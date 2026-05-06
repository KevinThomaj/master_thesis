
import gc
import os

import torch
import torchvision

from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from wilds import get_dataset

# Use sklearn's LabelEncoder for tabular dataframe columns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torchvision.transforms as T

class ClusteringPipeline:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_dataset(self):
        import pandas.core.tools.datetimes as _pd_dt_module
        _true_orig_to_datetime = _pd_dt_module.to_datetime

        def _patched_to_datetime(*args, **kwargs):
            kwargs['format'] = 'ISO8601'
            return _true_orig_to_datetime(*args, **kwargs)

        pd.to_datetime = _patched_to_datetime
        # never use dataset, but use the metadata(it contains all IMAGES)
        dataset = get_dataset(dataset="fmow", download = True)
        pd.to_datetime = _true_orig_to_datetime

        df = dataset.metadata.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
        df['year_extracted'] = df['timestamp'].dt.year

        region_names_list = dataset.metadata_map['region']
        df['region_names'] = df['region'].map(lambda x: region_names_list[int(x)])

        return df

    def sample_dataset(self, df, category_col='category', samples_per_class=32, random_state=42,target_year=2016,country_code="USA"):
        """Samples a balanced subset from the main dataframe."""

        #filtering based on year and state, just to simplify
        filtered_df = df[df['category'].isin(
            ["stadium","swimming_pool", "multi-unit_residential", "recreational_facility","park",
             "airport_hangar", "airport_terminal", "crop_field", "amusement_park", "military_facility",
             "nuclear_powerplant","airport", "runway","aquaculture", "wind_farm"])].copy()

        #filtered_df = df[df['category'].isin(["crop_field","airport_terminal"])]
        #filtered_df = df.copy()
        print(f"--- Sampling {samples_per_class} images per class ---")
        sample_indices = []

        for _, group in filtered_df.groupby(category_col):
            n_to_sample = min(len(group), samples_per_class)
            sampled_group = group.sample(n=n_to_sample, random_state=random_state)
            sample_indices.extend(sampled_group.index.tolist())

        df_sample = df.loc[sample_indices].copy()
        df_sample['index'] = sample_indices
        return df_sample

    def get_embeddings(self, df_sample, model, get_input_fn, transform_fn, batch_size=128,
                       save_path=None,compute_stats = True):
        """
        Extracts embeddings and creates a BRAND NEW, lean DataFrame
        that acts as your standalone dataset for clustering and training.
        """
        if save_path and os.path.exists(save_path):
            print(f"Found saved dataset at {save_path}. Loading from disk...")
            return pd.read_parquet(save_path)

        print("Extracting features...")
        model = model.to(self.device)
        model.eval()

        all_embeddings = []
        indices = df_sample['index'].tolist()


        #for mean and std
        channels_sum = torch.zeros(3).to(self.device)
        channels_squared_sum = torch.zeros(3).to(self.device)
        total_pixels = 0

        with torch.no_grad():
            for i in tqdm(range(0, len(indices), batch_size), desc="Extracting"):
                batch_idx = indices[i:i + batch_size]
                # 1. Load raw PIL images
                raw_imgs = [get_input_fn(idx) for idx in batch_idx]
                
                # 2. Compute Statistics on raw tensors (0-1 scaled, NO ImageNet normalization)
                if compute_stats:
                    raw_tensors = torch.stack([T.ToTensor()(img) for img in raw_imgs], dim=0).to(self.device)
                    B, C, H, W = raw_tensors.shape
                    channels_sum += torch.sum(raw_tensors, dim=[0, 2, 3])
                    channels_squared_sum += torch.sum(raw_tensors ** 2, dim=[0, 2, 3])
                    total_pixels += B * H * W
                    del raw_tensors # Free memory immediately

                # 3. Apply your full transform_fn (WITH ImageNet normalization) for the FM
                batch_imgs = [transform_fn(img) for img in raw_imgs]


                input_tensor = torch.stack(batch_imgs, dim=0).to(self.device)

                features = model(input_tensor)
                all_embeddings.append(features.cpu().numpy())

                # Strict Memory Cleanup
                del input_tensor, features, batch_imgs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        embeddings_matrix = np.concatenate(all_embeddings, axis=0)

        # ---------------------------------------------------------
        # CREATE THE NEW DATASET DATAFRAME
        # ---------------------------------------------------------
        df_dataset = pd.DataFrame({
            'original_index': indices,
            'category': df_sample['category'].values,
            'embedding': list(embeddings_matrix)
        })

        if save_path:
            print("Saving new embeddings dataset to disk...")
            df_dataset.to_parquet(save_path, engine='pyarrow')

        # 4. Finalize Statistics
        if compute_stats:
            mean = channels_sum / total_pixels
            std = (channels_squared_sum / total_pixels - mean ** 2) ** 0.5
            
            mean_list = mean.cpu().tolist()
            std_list = std.cpu().tolist()
            
            print(f"\nCalculated Dataset Mean: {mean_list}")
            print(f"Calculated Dataset Std:  {std_list}")
        return df_dataset

    def create_concepts(self, df,class_to_concept_mapping):
        # Modify the dataset dataframe to include the new labels
        df['concept'] = df['category'].map(class_to_concept_mapping)
        print("Distribution of images across the new macro-classes:")
        print(df['concept'].value_counts())
        return df

    def cluster_by_class(self, df, num_clusters=6, method='ward', metric='euclidean'):
        """Takes the unified dataframe, extracts embeddings natively, and performs clustering."""
        print("Clustering embeddings...")

        # Unstack the list of embeddings back into a 2D numpy array
        embeddings_matrix = np.stack(df['embedding'].values)

        df_emb = pd.DataFrame(embeddings_matrix)
        df_emb['category_name'] = df['category'].values

        class_centroids = df_emb.groupby('category_name').mean()
        name_classes = list(class_centroids.index)

        linkage_data = linkage(class_centroids, method=method, metric=metric)
        cluster_labels = fcluster(linkage_data, t=num_clusters, criterion='maxclust')

        class_to_macro_mapping = dict(zip(name_classes, cluster_labels))

        # Modify the dataset dataframe to include the new labels
        df['concept'] = df['category'].map(class_to_macro_mapping)

        print("Distribution of images across the new macro-classes:")
        print(df['concept'].value_counts())

        plt.figure(figsize=(20, 10))
        dendrogram(linkage_data, labels=name_classes, leaf_rotation=90, leaf_font_size=10)
        plt.title('Hierarchical Clustering Dendrogram', fontsize=16)
        plt.xlabel('Category Name', fontsize=14)
        plt.ylabel('Distance', fontsize=14)
        plt.tight_layout()
        plt.show()

        return df

    def train_offline(self, model, df, criterion, optimizer, target_col='category', epochs=100, batch_size=32,
                      log_interval=5):
        """Builds DataLoaders directly from the unified dataframe and trains the model."""
        print(f"Preparing data for training on target: {target_col}")

        # 1. Prepare Data
        X = np.stack(df['embedding'].values)
        le = LabelEncoder()
        y = le.fit_transform(df[target_col])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                torch.tensor(y_train, dtype=torch.long)),
                                  batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                               torch.tensor(y_test, dtype=torch.long)),
                                 batch_size=batch_size, shuffle=False)

        # 2. Train Loop
        model = model.to(self.device)
        history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

        for epoch in range(epochs):
            model.train()
            train_loss, correct, total = 0.0, 0, 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            train_acc = 100 * correct / total
            train_loss /= len(train_loader)

            val_loss, val_acc = self._evaluate(model, test_loader, criterion)

            history['train_loss'].append(train_loss)
            history['test_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(val_acc)

            if (epoch + 1) % log_interval == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

        print("\n--- Final Evaluation ---")
        final_test_loss, final_test_acc = self._evaluate(model, test_loader, criterion)
        print(f"Final Test Accuracy: {final_test_acc:.2f}%")
        return history

    def train_online(self, model, df, criterion, optimizer, get_input_fn, transform_fn, class_to_idx=None,
                     target_col='category', batch_size=32, num_epochs_per_batch=1,
                     distillator=None, distill_weight=1.0,ema = None):
        """
        Simulates an online datastream using Test-Then-Train (Prequential Evaluation).
        Supports optional Feature-based Knowledge Distillation.
        """
        # 1. Create a mapping from class names to integers if not provided
        if class_to_idx is None:
            unique_classes = df[target_col].unique()
            class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}

        model = model.to(self.device)

        history = {
            'total_samples_seen': [],
            'cumulative_accuracy': [],
            'drift_points': []
        }

        # 2. Inizializzazione Buffer e Metriche
        buffer_imgs = []
        buffer_labels = []
        buffer_teacher_features = []  # buffer for Feature Distillation

        concept_correct = 0
        concept_number_images = 0
        current_concept = 0
        total_samples_seen = 0

        # Iteriamo direttamente sul dataframe per simulare lo stream continuo
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Stream"):
            total_samples_seen += 1
            row_concept = row['concept']
            if row_concept != current_concept:
                print(
                    f"\n--- DRIFT DETECTED: Transitioning from Concept {current_concept} to Concept {row_concept} ---")
                print("Resetting accuracy metrics...")
                history['drift_points'].append(total_samples_seen)
                concept_correct = 0
                concept_number_images = 0
                current_concept = row_concept
                # Remember to reset EMA after a concept drift if there is EMA


            index = row['original_index']
            category = row[target_col]

            # --- 1. CARICAMENTO DATI REALI ---
            img_pil = get_input_fn(index)
            # Aggiungiamo dimensione batch e passiamo al device
            img_tensor = transform_fn(img_pil).unsqueeze(0).to(self.device)

            # Convertiamo l'etichetta stringa nel numero intero corrispondente
            label_idx = class_to_idx[category]
            label_tensor = torch.tensor([label_idx], dtype=torch.long).to(self.device)

            # Se la distillazione è attiva, estraiamo l'embedding del teacher dal dataframe
            if distillator is not None:
                teacher_feat = torch.tensor(row['embedding'], dtype=torch.float32).unsqueeze(0).to(self.device)
                buffer_teacher_features.append(teacher_feat)

            # --- 2. PREQUENTIAL EVALUATION (TEST) ---
            if distillator is not None:
                distillator.eval()
            if ema is not None:
                prediction_logits = ema.predict(img_tensor)
                predicted_class = torch.argmax(prediction_logits, dim=1)

                # Aggiorniamo le metriche
                if predicted_class.item() == label_idx:
                    concept_correct += 1
                concept_number_images += 1

            else:
                model.eval()
                with torch.no_grad():
                    # Il modello modificato restituisce un dict, estraiamo i 'logits'
                    prediction_output = model(img_tensor)
                    prediction_logits = prediction_output['logits']

                    predicted_class = torch.argmax(prediction_logits, dim=1)

                    # Aggiorniamo le metriche
                    if predicted_class.item() == label_idx:
                        concept_correct += 1
                    concept_number_images += 1


            # --- 3. AGGIUNTA AL BUFFER ---
            buffer_imgs.append(img_tensor)
            buffer_labels.append(label_tensor)

            # --- 4. ONLINE LEARNING (TRAIN SUL BATCH) ---
            if len(buffer_imgs) == batch_size:
                model.train()  # Modalità allenamento
                if distillator is not None:
                    distillator.train()

                batch_imgs = torch.cat(buffer_imgs, dim=0)
                batch_labels = torch.cat(buffer_labels, dim=0)

                if distillator is not None:
                    batch_teacher = torch.cat(buffer_teacher_features, dim=0)

                loss_ce_history = []
                loss_distill_history = []
                for epoch in range(num_epochs_per_batch):
                    optimizer.zero_grad()

                    # Forward pass sull'intero batch
                    student_output = model(batch_imgs)
                    logits = student_output['logits']

                    # Calcolo Loss per il Task di Classificazione (CrossEntropy)
                    loss_ce = criterion(logits, batch_labels)
                    total_loss = loss_ce
                    loss_ce_history.append(loss_ce.item())
                    # Calcolo Loss per il Task di Distillazione
                    if distillator is not None:
                        student_features = student_output['features']
                        loss_distill = distillator(student_features, batch_teacher)
                        #DA RIMUOVERE OCCHIO il coefficiente 0
                        # Combiniamo le loss
                        total_loss = 0.0* loss_ce + (distill_weight * loss_distill)

                        loss_distill_history.append(loss_distill.item())

                    # Backward pass basato sulla loss totale
                    total_loss.backward()
                    optimizer.step()

                #Update EMA if needed
                if ema is not None:
                    ema.update(model)
                # --- 5. POPULATE HISTORY ---
                acc_corrente = (concept_correct / concept_number_images) * 100 if concept_number_images > 0 else 0
                history['total_samples_seen'].append(total_samples_seen)
                history['cumulative_accuracy'].append(acc_corrente)

                print(f"Loss output student: {np.mean(loss_ce_history):.4f}")
                if distillator is not None:
                    print(f"Loss distillation: {np.mean(loss_distill_history):.4f}")
                print(
                    f"Sample visti: {concept_number_images} | Classi corrette: {concept_correct} | Accuratezza Cumulata: {acc_corrente:.2f}%")
                #Clear buffer for next cycle
                buffer_imgs.clear()
                buffer_labels.clear()
                if distillator is not None:
                    buffer_teacher_features.clear()

        # Optional: Print final accuracy at the very end of the stream
        final_acc = (concept_correct / concept_number_images) * 100 if concept_number_images > 0 else 0
        print(f"\n--- Fine Stream | Accuratezza Finale: {final_acc:.2f}% ---")

        return final_acc,history

    def linear_probing_online(self, classifier, df, criterion, optimizer, class_to_idx=None,
                              target_col='category', batch_size=32, num_epochs_per_batch=1):
        """
        Simulates an online datastream using Test-Then-Train for Linear Probing.
        Uses pre-computed embeddings directly from the dataframe.
        """
        print("--- Avvio Streaming Pipeline (Linear Probing Online su Embeddings Pre-calcolati) ---")

        # 1. Mappatura classi
        if class_to_idx is None:
            unique_classes = df[target_col].unique()
            class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}

        # 2. Configurazione Modello
        classifier = classifier.to(self.device)

        # 3. Inizializzazione Buffer
        buffer_features = []
        buffer_labels = []

        # 1. Initialize History Tracking
        history = {
            'total_samples_seen': [],
            'cumulative_accuracy': [],
        }

        concept_correct = 0
        concept_number_images = 0
        current_concept = 0
        total_samples_seen = 0

        # Iteriamo sul dataframe per simulare lo stream
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Stream"):
            total_samples_seen += 1

            row_concept = row['concept']
            if row_concept != current_concept:
                print(
                    f"\n--- DRIFT DETECTED: Transitioning from Concept {current_concept} to Concept {row_concept} ---")
                print("Resetting accuracy metrics...")
                concept_correct = 0
                concept_number_images = 0
                current_concept = row_concept
            category = row[target_col]

            # --- 1. CARICAMENTO FEATURES PRE-CALCOLATE ---
            # L'embedding è già un numpy array nel dataframe. Lo convertiamo in tensor.
            # Usiamo unsqueeze(0) per aggiungere la dimensione del batch (1, embedding_dim)
            features = torch.tensor(row['embedding'], dtype=torch.float32).unsqueeze(0).to(self.device)

            label_idx = class_to_idx[category]
            label_tensor = torch.tensor([label_idx], dtype=torch.long).to(self.device)

            # --- 2. PREQUENTIAL EVALUATION (TEST SUL CLASSIFICATORE) ---
            classifier.eval()
            with torch.no_grad():
                prediction_logits = classifier(features)
                predicted_class = torch.argmax(prediction_logits, dim=1)

                # Aggiorniamo le metriche
                if predicted_class.item() == label_idx:
                    concept_correct += 1
                concept_number_images += 1

            # --- 3. AGGIUNTA AL BUFFER ---
            buffer_features.append(features)
            buffer_labels.append(label_tensor)

            # --- 4. ONLINE LEARNING (TRAIN DEL SOLO CLASSIFICATORE SUL BATCH) ---
            if len(buffer_features) == batch_size:
                classifier.train()

                batch_features = torch.cat(buffer_features, dim=0)
                batch_labels = torch.cat(buffer_labels, dim=0)
                loss_ce_history = []
                for epoch in range(num_epochs_per_batch):
                    optimizer.zero_grad()

                    # Forward pass solo del classificatore
                    logits = classifier(batch_features)

                    # Calcolo Loss e Backward
                    loss_ce = criterion(logits, batch_labels)
                    loss_ce.backward()
                    optimizer.step()
                    loss_ce_history.append(loss_ce.item())

                # --- 2. POPULATE HISTORY ---
                acc_corrente = (concept_correct / concept_number_images) * 100
                history['total_samples_seen'].append(total_samples_seen)
                history['cumulative_accuracy'].append(acc_corrente)

                print(
                    f"Sample visti: {concept_number_images} | Classi corrette: {concept_correct} | Accuratezza Cumulata: {acc_corrente:.2f}%")
                # Svuotiamo i buffer per il prossimo ciclo
                buffer_features.clear()
                buffer_labels.clear()

        # Print finale
        final_acc = (concept_correct / concept_number_images) * 100 if concept_number_images > 0 else 0
        print(f"\n--- Fine Stream | Accuratezza Finale Linear Probing: {final_acc:.2f}% ---")
        return final_acc,history



    def _evaluate(self, model, loader, criterion):
        """Internal helper for evaluation."""
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                running_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        return running_loss / len(loader), 100 * correct / total
