
import gc
import os

import torch

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

class ClusteringPipeline:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_dataset(self, path="/mnt/windows/Users/kevin/Downloads/"):
        import pandas.core.tools.datetimes as _pd_dt_module
        _true_orig_to_datetime = _pd_dt_module.to_datetime

        def _patched_to_datetime(*args, **kwargs):
            kwargs['format'] = 'ISO8601'
            return _true_orig_to_datetime(*args, **kwargs)

        pd.to_datetime = _patched_to_datetime
        # never use dataset, but use the metadata(it contains all IMAGES)
        dataset = get_dataset(dataset="fmow", root_dir=path, download=False)
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
        filtered_df = df[(df['year_extracted'] == target_year) & (df['country_code'] == country_code)
            & df['category'].isin(["crop_field","airport_terminal","recreational_facility","place_of_worship","stadium"])].copy()
        #filtered_df = df.copy()
        #filtered_df = df[df['category'].isin(["crop_field","airport_terminal"])]
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
                       save_path=None):
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

        with torch.no_grad():
            for i in tqdm(range(0, len(indices), batch_size), desc="Extracting"):
                batch_idx = indices[i:i + batch_size]
                batch_imgs = [transform_fn(get_input_fn(idx)) for idx in batch_idx]
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

        return df_dataset

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
        df['macro_class'] = df['category'].map(class_to_macro_mapping)

        print("Distribution of images across the new macro-classes:")
        print(df['macro_class'].value_counts())

        plt.figure(figsize=(20, 10))
        dendrogram(linkage_data, labels=name_classes, leaf_rotation=90, leaf_font_size=10)
        plt.title('Hierarchical Clustering Dendrogram', fontsize=16)
        plt.xlabel('Category Name', fontsize=14)
        plt.ylabel('Distance', fontsize=14)
        plt.tight_layout()
        plt.show()

        return df

    def visualize_embedding_space(self, df, color_by='category', method='tsne', random_state=42):
        """
        Visualizes the high-dimensional embedding space by reducing it to 2D.

        Args:
            df (pd.DataFrame): The dataframe containing the 'embedding' column.
            color_by (str): The column name used to color the points (e.g., 'category' or 'macro_class').
            method (str): The dimensionality reduction technique ('tsne' or 'pca').
            random_state (int): Random seed for reproducibility.
        """

        print(f"--- Reducing dimensionality using {method.upper()} ---")

        # Unstack embeddings back into a 2D array
        embeddings_matrix = np.stack(df['embedding'].values)

        # Select and fit the dimensionality reduction model
        if method.lower() == 'tsne':
            # t-SNE is generally better for capturing non-linear local relationships in embeddings
            reducer = TSNE(n_components=2, random_state=random_state, init='pca', learning_rate='auto')
        elif method.lower() == 'pca':
            # PCA is much faster, good for a quick global view
            reducer = PCA(n_components=2, random_state=random_state)
        else:
            raise ValueError("Method must be 'tsne' or 'pca'.")

        reduced_embeddings = reducer.fit_transform(embeddings_matrix)

        # Create the plot
        plt.figure(figsize=(14, 10))
        unique_labels = df[color_by].unique()

        # Use a colormap with enough distinct colors (tab20 supports up to 20 distinct colors)
        cmap = plt.get_cmap('tab20')

        # Scatter plot for each unique label to build the legend properly
        for i, label in enumerate(unique_labels):
            # Find the indices where the dataframe matches the current label
            idx = df[color_by] == label

            plt.scatter(
                reduced_embeddings[idx, 0],
                reduced_embeddings[idx, 1],
                label=label,
                alpha=0.7,
                c=[cmap(i % 20)],  # Loop back through colors if more than 20 classes
                edgecolors='w',
                linewidth=0.5
            )

        plt.title(f"2D Embedding Space Visualization ({method.upper()})", fontsize=16)
        plt.xlabel("Component 1", fontsize=12)
        plt.ylabel("Component 2", fontsize=12)

        # Place legend outside the plot so it doesn't overlap the data
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, markerscale=1.5)

        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()


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
                     target_col='category', batch_size=32, num_epochs_per_batch=1, shuffle=True,order_by_concept = False,
                     distillator=None, distill_weight=1.0):
        """
        Simulates an online datastream using Test-Then-Train (Prequential Evaluation).
        Supports optional Feature-based Knowledge Distillation.
        """
        print("--- Avvio Streaming Pipeline (Prequential Evaluation) ---")
        if shuffle:
            print("Shuffling the dataset before streaming...")
            # frac=1 returns all rows in random order.
            # reset_index(drop=True) creates a clean, sequential index for the new order.
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        if order_by_concept:
            df = df.sort_values(by='macro_class').reset_index(drop=True)


        # 1. Create a mapping from class names to integers if not provided
        if class_to_idx is None:
            unique_classes = df[target_col].unique()
            class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}

        model = model.to(self.device)


        # 2. Inizializzazione Buffer e Metriche
        buffer_imgs = []
        buffer_labels = []
        buffer_teacher_features = []  # buffer for Feature Distillation

        concept_corrette = 0
        concept_viste = 0
        concept_corrente = -1

        # Iteriamo direttamente sul dataframe per simulare lo stream continuo
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Stream"):

            if order_by_concept:
                # Se il concept cambia, stampiamo un avviso (Simulazione Concept Drift!)
                if row['macro_class'] != concept_corrente:
                    if concept_corrente != -1 and concept_viste > 0:
                        acc_finale = (concept_corrette / concept_viste) * 100
                        print(f"\n---> [FINE CONCEPT {concept_corrente}] Accuratezza Chiusura: {acc_finale:.2f}% <---")
                    # Rilevato nuovo Concept: Reset dei contatori!
                    print(f"\n[!] CONCEPT DRIFT: Inizio Concept {row['macro_class']}")
                    concept_corrente = row['macro_class']
                    concept_corrette = 0
                    concept_viste = 0


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
            model.eval()
            if distillator is not None:
                distillator.eval()

            with torch.no_grad():
                # Il modello modificato restituisce un dict, estraiamo i 'logits'
                prediction_output = model(img_tensor)
                prediction_logits = prediction_output['logits']

                predicted_class = torch.argmax(prediction_logits, dim=1)

                # Aggiorniamo le metriche
                if predicted_class.item() == label_idx:
                    concept_corrette += 1
                concept_viste += 1

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

                        # Combiniamo le loss
                        total_loss = loss_ce + (distill_weight * loss_distill)

                        loss_distill_history.append(loss_distill.item())

                    # Backward pass basato sulla loss totale
                    total_loss.backward()
                    optimizer.step()
                print(f"Loss output student: {np.mean(loss_ce_history):.4f}")
                if distillator is not None:
                    print(f"Loss distillation: {np.mean(loss_distill_history):.4f}")

                # Stampa dell'accuratezza cumulata
                acc_corrente = (concept_corrette / concept_viste) * 100
                print(
                    f"Sample visti: {concept_viste} | Classi corrette: {concept_corrette} | Accuratezza Cumulata: {acc_corrente:.2f}%")

                # Svuotiamo i buffer per il prossimo ciclo
                buffer_imgs.clear()
                buffer_labels.clear()
                if distillator is not None:
                    buffer_teacher_features.clear()

        # Optional: Print final accuracy at the very end of the stream
        final_acc = (concept_corrette / concept_viste) * 100 if concept_viste > 0 else 0
        print(f"\n--- Fine Stream | Accuratezza Finale: {final_acc:.2f}% ---")

    def linear_probing_online(self, classifier, df, criterion, optimizer, class_to_idx=None,
                              target_col='category', batch_size=32, num_epochs_per_batch=1, shuffle=True):
        """
        Simulates an online datastream using Test-Then-Train for Linear Probing.
        Uses pre-computed embeddings directly from the dataframe.
        """
        print("--- Avvio Streaming Pipeline (Linear Probing Online su Embeddings Pre-calcolati) ---")

        if shuffle:
            print("Shuffling the dataset before streaming...")
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # 1. Mappatura classi
        if class_to_idx is None:
            unique_classes = df[target_col].unique()
            class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}

        # 2. Configurazione Modello
        classifier = classifier.to(self.device)

        # 3. Inizializzazione Buffer
        buffer_features = []
        buffer_labels = []

        concept_corrette = 0
        concept_viste = 0

        # Iteriamo sul dataframe per simulare lo stream
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Stream"):
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
                    concept_corrette += 1
                concept_viste += 1

            # --- 3. AGGIUNTA AL BUFFER ---
            buffer_features.append(features)
            buffer_labels.append(label_tensor)

            # --- 4. ONLINE LEARNING (TRAIN DEL SOLO CLASSIFICATORE SUL BATCH) ---
            if len(buffer_features) == batch_size:
                classifier.train()

                batch_features = torch.cat(buffer_features, dim=0)
                batch_labels = torch.cat(buffer_labels, dim=0)

                for epoch in range(num_epochs_per_batch):
                    optimizer.zero_grad()

                    # Forward pass solo del classificatore (istantaneo)
                    logits = classifier(batch_features)

                    # Calcolo Loss e Backward
                    loss_ce = criterion(logits, batch_labels)
                    loss_ce.backward()
                    optimizer.step()

                # Stampa dell'accuratezza cumulata
                acc_corrente = (concept_corrette / concept_viste) * 100
                print(
                    f"Sample visti: {concept_viste} | Classi corrette: {concept_corrette} | Accuratezza Cumulata: {acc_corrente:.2f}%")

                # Svuotiamo i buffer per il prossimo ciclo
                buffer_features.clear()
                buffer_labels.clear()

        # Print finale
        final_acc = (concept_corrette / concept_viste) * 100 if concept_viste > 0 else 0
        print(f"\n--- Fine Stream | Accuratezza Finale Linear Probing: {final_acc:.2f}% ---")



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



    def visualize_category_images(self,df, category_name, num_images=5, get_input_fn = None):
        """
        Visualizes a specified number of images from a chosen category.

        Args:
            df: The dataframe containing 'category' and 'index' columns.
            category_name: The string name of the category to visualize.
            num_images: The number of images to display.
            get_input_fn: The function used to load the image based on its index.
        """
        # Filter dataframe for the requested category
        subset = df[df['category'] == category_name]

        if subset.empty:
            print(f"Error: Category '{category_name}' not found in the dataframe.")
            return

        # Ensure we don't request more images than available
        actual_num_images = min(len(subset), num_images)
        if actual_num_images < num_images:
            print(f"Note: Only {actual_num_images} images available for '{category_name}'.")

        # Randomly sample the indices
        sampled_indices = subset.sample(n=actual_num_images, random_state=42)['index'].tolist()

        # Calculate grid dimensions (max 5 columns for readability)
        cols = min(actual_num_images, 5)
        rows = math.ceil(actual_num_images / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        fig.suptitle(f"Category: {category_name} ({actual_num_images} images)", fontsize=16)

        # Flatten axes array for easy iteration, handling 1D and 2D arrays
        if actual_num_images == 1:
            axes = [axes]
        elif rows > 1 or cols > 1:
            axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i < actual_num_images:
                idx = sampled_indices[i]
                img = get_input_fn(idx)  # Uses your existing image loading logic
                ax.imshow(img)
                ax.set_title(f"Idx: {idx}")
                ax.axis('off')
            else:
                # Hide empty subplots if the grid isn't perfectly filled
                ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)  # Adjust title spacing
        plt.show()
