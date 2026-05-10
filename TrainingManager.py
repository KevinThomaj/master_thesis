import copy

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from FmowTorchDataset import FmowTorchDataset


class TrainingManager:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def pretrain_teacher(self):
        pass

    def pretrain_student(
            self,
            student,
            df_train,
            manager,
            class_to_idx,
            transform_fn,
            use_embeddings=False,
            epochs=50,
            batch_size=64,
            lr=1e-3,
            patience=5,
            device='cuda',
            projector=None,
            lambda_distill=1.0
    ):
        """
        Unified pretraining loop that handles both standard pretraining and
        offline feature distillation dynamically.
        """
        print(f"\n--- Initializing Pretraining (Distillation/Embeddings: {use_embeddings}) ---")

        # 1. Stratified Train/Val Split (80% Train, 20% Val)
        train_df, val_df = train_test_split(
            df_train, test_size=0.2, stratify=df_train['category'], random_state=42
        )
        print(f"Training on {len(train_df)} samples, Validating on {len(val_df)} samples.")

        # 2. Setup DataLoaders with the unified dataset
        train_dataset = FmowTorchDataset(train_df, manager, class_to_idx, transform_fn, use_embeddings)
        val_dataset = FmowTorchDataset(val_df, manager, class_to_idx, transform_fn, use_embeddings)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # 3. Setup Optimizers & Devices
        criterion_ce = nn.CrossEntropyLoss()
        student = student.to(device)

        trainable_params = list(student.parameters())
        if projector is not None:
            projector = projector.to(device)
            trainable_params += list(projector.parameters())

        optimizer = optim.Adam(trainable_params, lr=lr)

        # 4. Early Stopping Trackers
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_student_wts = copy.deepcopy(student.state_dict())
        best_projector_wts = copy.deepcopy(projector.state_dict())

        # 5. Training Loop
        for epoch in range(epochs):
            student.train()
            if projector: projector.train()

            train_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
                optimizer.zero_grad()

                # --- DYNAMIC UNPACKING ---
                if use_embeddings:
                    imgs, teacher_features, labels = batch
                    teacher_features = teacher_features.to(device)
                else:
                    imgs, labels = batch

                imgs, labels = imgs.to(device), labels.to(device)

                # Forward Student
                outputs = student(imgs)
                logits = outputs['logits']
                features_stud = outputs['features']

                # Standard Cross Entropy Loss
                loss_ce = criterion_ce(logits, labels)

                # Distillation Loss (Conditional)
                if use_embeddings and projector is not None:
                    loss_distill = projector(features_stud, teacher_features)
                    loss = loss_ce + (lambda_distill * loss_distill)
                else:
                    loss = loss_ce

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * imgs.size(0)

            train_loss /= len(train_loader.dataset)

            # 6. Validation Phase
            student.eval()
            if projector: projector.eval()

            val_loss = 0.0
            correct = 0

            with torch.no_grad():
                for batch in val_loader:
                    # --- DYNAMIC UNPACKING ---
                    if use_embeddings:
                        imgs, teacher_features, labels = batch
                        teacher_features = teacher_features.to(device)
                    else:
                        imgs, labels = batch

                    imgs, labels = imgs.to(device), labels.to(device)

                    outputs = student(imgs)
                    logits = outputs['logits']
                    features_stud = outputs['features']

                    loss_ce = criterion_ce(logits, labels)

                    if use_embeddings and projector is not None:
                        loss_distill = projector(features_stud, teacher_features)
                        loss = loss_ce + (lambda_distill * loss_distill)
                    else:
                        loss = loss_ce

                    val_loss += loss.item() * imgs.size(0)
                    preds = torch.argmax(logits, dim=1)
                    correct += torch.sum(preds == labels.data)

            val_loss /= len(val_loader.dataset)
            val_acc = correct.double() / len(val_loader.dataset)

            print(
                f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # 7. Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_student_wts = copy.deepcopy(student.state_dict())
                epochs_no_improve = 0
                if projector is not None:
                    best_projector_wts = copy.deepcopy(projector.state_dict())
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered! No improvement for {patience} epochs.")
                    break

        print("Training complete. Restoring best model weights.")
        student.load_state_dict(best_student_wts)
        if projector is not None:
            projector.load_state_dict(best_projector_wts)
            return student, projector

        return student, None

    def train_online(self,
                     model,
                     df,
                     criterion,
                     optimizer,
                     get_input_fn,
                     transform_fn,
                     class_to_idx,
                     target_col='category',
                     batch_size=50,
                     num_epochs_per_batch=1,
                     distillator=None,
                     distill_weight=1.0,
                     ema=None,
                     inference_only=False):
        """
        Gestisce dinamicamente tutte le 5 varianti di stream:
        Inference Only, Standard Fine-tuning, Feature Distillation, ed EMA.
        """

        model = model.to(self.device)

        history = {
            'total_samples_seen': [],
            'cumulative_accuracy': [],
            'drift_points': []
        }

        buffer_imgs = []
        buffer_labels = []
        buffer_teacher_features = []

        concept_correct = 0
        concept_number_images = 0
        current_concept = 0
        total_samples_seen = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Stream (Inference Only: {inference_only})"):
            total_samples_seen += 1
            row_concept = row['concept']

            # --- DRIFT DETECTION ---
            if row_concept != current_concept:
                print(f"\n--- DRIFT DETECTED: Transition da Concept {current_concept} a Concept {row_concept} ---")
                history['drift_points'].append(total_samples_seen)
                concept_correct = 0
                concept_number_images = 0
                current_concept = row_concept

            index = row['image_index']
            category = row[target_col]

            # --- 1. CARICAMENTO DATI ---
            img_pil = get_input_fn(index)
            img_tensor = transform_fn(img_pil).unsqueeze(0).to(self.device)

            label_idx = class_to_idx[category]
            label_tensor = torch.tensor([label_idx], dtype=torch.long).to(self.device)

            # Estraiamo l'embedding del teacher SOLO se stiamo facendo distillazione attiva
            if distillator is not None and not inference_only:
                teacher_feat = torch.tensor(row['embedding'], dtype=torch.float32).unsqueeze(0).to(self.device)
                buffer_teacher_features.append(teacher_feat)

            # --- 2. PREQUENTIAL EVALUATION (TEST FASE) ---
            if distillator is not None:
                distillator.eval()

            if ema is not None:
                prediction_logits = ema.predict(img_tensor)  # Usa la logica predict del tuo EMA wrapper
                predicted_class = torch.argmax(prediction_logits, dim=1)
            else:
                model.eval()
                with torch.no_grad():
                    prediction_output = model(img_tensor)
                    prediction_logits = prediction_output['logits']
                    predicted_class = torch.argmax(prediction_logits, dim=1)

            if predicted_class.item() == label_idx:
                concept_correct += 1
            concept_number_images += 1

            acc_corrente = (concept_correct / concept_number_images) * 100 if concept_number_images > 0 else 0

            # --- 3. BIVIO INFERENZA VS TRAINING ---
            history['total_samples_seen'].append(total_samples_seen)
            history['cumulative_accuracy'].append(acc_corrente)
            if inference_only:
                continue
            # --- 4. BUFFERING E TRAINING (ONLINE LEARNING) ---
            buffer_imgs.append(img_tensor)
            buffer_labels.append(label_tensor)

            if len(buffer_imgs) == batch_size:
                model.train()
                if distillator is not None:
                    distillator.train()

                batch_imgs = torch.cat(buffer_imgs, dim=0)
                batch_labels = torch.cat(buffer_labels, dim=0)
                if distillator is not None:
                    batch_teacher = torch.cat(buffer_teacher_features, dim=0)

                for epoch in range(num_epochs_per_batch):
                    optimizer.zero_grad()

                    student_output = model(batch_imgs)
                    logits = student_output['logits']

                    loss_ce = criterion(logits, batch_labels)
                    total_loss = loss_ce

                    if distillator is not None:
                        student_features = student_output['features']
                        loss_distill = distillator(student_features, batch_teacher)
                        total_loss = loss_ce + (distill_weight * loss_distill)

                    total_loss.backward()
                    optimizer.step()

                # Update dell'EMA dopo che i pesi del modello principale sono stati aggiornati
                if ema is not None:
                    ema.update(model)

                # Svuota i buffer
                buffer_imgs.clear()
                buffer_labels.clear()
                if distillator is not None:
                    buffer_teacher_features.clear()

        final_acc = (concept_correct / concept_number_images) * 100 if concept_number_images > 0 else 0
        print(f"\n--- Fine Stream | Accuratezza Finale Concept {current_concept}: {final_acc:.2f}% ---")

        return final_acc, history