import copy
from lightly.data import DINOCollateFunction
from lightly.loss import DINOLoss
from lightly.models.utils import update_momentum
from torch.cuda.amp import GradScaler, autocast
import os

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

    def pretrain_teacher(
            self,
            foundation_model,
            df,
            manager,
            save_path="./pretrained_dinov2_lora.pth",
            epochs=25,
            batch_size=32,  # Local batch size
            accumulation_steps=8,  # Effective batch size = 32 * 8 = 256
            lr=1e-3,
    ):
        print("\n--- Starting Extended DINOv2 Pre-training with LoRA ---")

        from FoundationModel import DINOPretrainWrapper
        dino_wrapper = DINOPretrainWrapper(foundation_model).to(self.device)

        # 1. Multi-Crop Augmentation Pipeline (Lightly)
        # Yields 2 global crops (224x224) and 6 local crops (98x98)
        # 1. Multi-Crop Augmentation Pipeline (Lightly)
        lightly_collate = DINOCollateFunction(
            global_crop_size=224,
            global_crop_scale=(0.4, 1.0),
            local_crop_size=98,
            local_crop_scale=(0.05, 0.4),
            n_local_views=6
        )

        # --- 🚨 ADAPTER FIX: Wrap the collate function ---
        def custom_collate(batch):
            # FmowTorchDataset returns a list of 2-tuples: (image, label)
            # Lightly expects 3-tuples: (image, label, filename)
            # We append a dummy string to each item to prevent the crash
            adapted_batch = [(img, label, "dummy_file") for img, label in batch]
            return lightly_collate(adapted_batch)

        # -------------------------------------------------

        # 2. Dataset Setup
        # We pass transform=None because DINOCollateFunction expects raw PIL images
        pretrain_dataset = FmowTorchDataset(
            df=df,
            manager=manager,
            class_to_idx={cat: 0 for cat in df['category'].unique()},
            transform=None,
            use_embeddings=False
        )

        data_loader = DataLoader(
            pretrain_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate,
            num_workers=8,
            pin_memory=True,
            drop_last=True
        )

        # 3. Loss and Optimizer
        criterion = DINOLoss(
            output_dim=65536,
            warmup_teacher_temp_epochs=10,  # Crucial for DINO stability
        ).to(self.device)

        # Only pass parameters that require gradients (LoRA + Student Head)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, dino_wrapper.parameters()), #only student parameters
            lr=lr,
            weight_decay=1e-4
        )

        # 4. Training Loop
        for epoch in range(epochs):
            dino_wrapper.train()
            total_loss = 0.0

            # Update DINO Loss epoch (for temperature scheduling)
            criterion.epoch = epoch

            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs} [DINO SSL]")
            optimizer.zero_grad()

            for step, batch in enumerate(progress_bar):
                # Lightly collate_fn returns: (list_of_views, labels, filenames)
                views = batch[0]
                views = [view.to(self.device) for view in views]
                global_views = views[:2]

                # --- BFloat16 Mixed Precision for Ada Lovelace ---
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,enabled=(self.device.type == 'cuda')):
                    # Teacher process (Global views only)
                    with torch.no_grad():
                        teacher_out = [dino_wrapper.forward_teacher(view) for view in global_views]

                    # Student process (All views)
                    student_out = [dino_wrapper(view) for view in views]

                    # Loss calculation
                    loss = criterion(student_out, teacher_out) / accumulation_steps

                # Backward pass
                loss.backward()
                total_loss += loss.item() * accumulation_steps

                # Gradient Accumulation & EMA Update
                if ((step + 1) % accumulation_steps == 0) or ((step + 1) == len(data_loader)):
                    optimizer.step()
                    optimizer.zero_grad()

                    # EMA Update for Teacher
                    # Momentum usually scales from 0.996 to 1.0 during training
                    update_momentum(
                        dino_wrapper.student_backbone, dino_wrapper.teacher_backbone, m=0.996
                    )
                    update_momentum(
                        dino_wrapper.student_head, dino_wrapper.teacher_head, m=0.996
                    )

                progress_bar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})

            avg_loss = total_loss / len(data_loader)
            print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

        # 5. Save the LoRA weights cleanly
        print(f"Saving Extended Foundation Model to {save_path}...")
        # Save only the state dict of the base foundation model (which includes peft LoRA)
        torch.save(foundation_model.state_dict(), save_path)

    def pretrain_student(
            self,
            student,
            df,
            manager,
            class_to_idx,
            transform_fn,
            use_embeddings=False,
            epochs=50,
            batch_size=64,
            #let's try to decrease learning rate to 1e-4
            lr=1e-4,
            patience=5,
            projector=None,
            lambda_distill=1.0
    ):
        """
        Unified pretraining loop that handles both standard pretraining and
        offline feature distillation dynamically.
        """
        print(f"\n--- Initializing Pretraining (Distillation/Embeddings: {use_embeddings}) ---")

        # 1. Stratified Train/Val/Test Split (80% Train, 10% Val, 10% Test)
        # First split: 80% Train, 20% Temp (which will be split into Val and Test)
        train_df, temp_df = train_test_split(
            df, test_size=0.2, stratify=df['category'], random_state=42
        )

        # Second split: 50% of the Temp goes to Validation, 50% to Test (i.e. 10% / 10% of total)
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df['category'], random_state=42
        )

        print(
            f"Training on {len(train_df)} samples, Validating on {len(val_df)} samples, Testing on {len(test_df)} samples.")

        # 2. Setup DataLoaders with the unified dataset
        train_dataset = FmowTorchDataset(train_df, manager, class_to_idx, transform_fn, use_embeddings)
        val_dataset = FmowTorchDataset(val_df, manager, class_to_idx, transform_fn, use_embeddings)
        test_dataset = FmowTorchDataset(test_df, manager, class_to_idx, transform_fn, use_embeddings)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # 3. Setup Optimizers & Devices
        criterion_ce = nn.CrossEntropyLoss()
        student = student.to(self.device)

        trainable_params = list(student.parameters())
        if projector is not None:
            projector = projector.to(self.device)
            trainable_params += list(projector.parameters())

        optimizer = optim.Adam(trainable_params, lr=lr)

        # 4. Early Stopping Trackers
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_student_wts = copy.deepcopy(student.state_dict())
        best_projector_wts = copy.deepcopy(projector.state_dict()) if projector else None

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
                    teacher_features = teacher_features.to(self.device)
                else:
                    imgs, labels = batch

                imgs, labels = imgs.to(self.device), labels.to(self.device)

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
                        teacher_features = teacher_features.to(self.device)
                    else:
                        imgs, labels = batch

                    imgs, labels = imgs.to(self.device), labels.to(self.device)

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

        # ------------------------------------------------------------------
        # 8. TEST PHASE (Extracted from df)
        # ------------------------------------------------------------------
        print("\n--- Evaluating on Test Set ---")
        student.eval()
        if projector: projector.eval()

        test_loss = 0.0
        test_correct = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                if use_embeddings:
                    imgs, teacher_features, labels = batch
                    teacher_features = teacher_features.to(self.device)
                else:
                    imgs, labels = batch

                imgs, labels = imgs.to(self.device), labels.to(self.device)

                outputs = student(imgs)
                logits = outputs['logits']
                features_stud = outputs['features']

                loss_ce = criterion_ce(logits, labels)

                if use_embeddings and projector is not None:
                    loss_distill = projector(features_stud, teacher_features)
                    loss = loss_ce + (lambda_distill * loss_distill)
                else:
                    loss = loss_ce

                test_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(logits, dim=1)
                test_correct += torch.sum(preds == labels.data)

        test_loss /= len(test_loader.dataset)
        test_acc = test_correct.double() / len(test_loader.dataset)

        print(f"Final Test Loss: {test_loss:.4f} | Final Test Acc: {test_acc:.4f}")
        # ------------------------------------------------------------------

        if projector is not None:
            return student, projector

        return student, None

    def train_online(self,
                     model,
                     df,
                     criterion,
                     optimizer,
                     manager,  # Need to pass manager for FmowTorchDataset
                     transform_fn,
                     class_to_idx,
                     target_col='category',
                     batch_size=25,
                     num_epochs_per_batch=1,
                     distillator=None,
                     distill_weight=1.0,
                     ema=None,
                     inference_only=False):
        """
        Ottimizzato con DataLoader (I/O asincrono), inferenza batched su GPU,
        e calcolo dell'accuratezza globale.
        """

        # 1. SETUP DATALOADER
        use_emb = (distillator is not None and not inference_only)
        stream_dataset = FmowTorchDataset(
            df=df,
            manager=manager,
            class_to_idx=class_to_idx,
            transform=transform_fn,
            use_embeddings=use_emb
        )

        # num_workers=4 will fetch images from disk in the background while the GPU trains
        stream_loader = DataLoader(
            stream_dataset,
            batch_size=batch_size,
            shuffle=False,  # MUST BE FALSE to maintain temporal stream order
            num_workers=4,
            pin_memory=True
        )

        model = model.to(self.device)

        history = {
            'total_samples_seen': [],
            'cumulative_accuracy': [],
            'drift_points': []
        }

        # Counters
        global_correct = 0
        global_total = 0
        concept_correct = 0
        concept_number_images = 0
        current_concept = None
        total_samples_seen = 0

        # Iterate over the DataLoader
        for batch in tqdm(stream_loader, desc=f"Stream (Inference Only: {inference_only})"):

            # --- 2. UNPACK BATCH ---
            if use_emb:
                batch_imgs, batch_teacher, batch_labels = batch
                batch_teacher = batch_teacher.to(self.device)
            else:
                batch_imgs, batch_labels = batch

            batch_imgs = batch_imgs.to(self.device)
            batch_labels = batch_labels.to(self.device)

            # --- 3. PREQUENTIAL EVALUATION (TEST FASE BATCHED) ---
            if distillator is not None:
                distillator.eval()

            if ema is not None:
                with torch.no_grad():
                    prediction_logits = ema.predict(batch_imgs)
            else:
                model.eval()
                with torch.no_grad():
                    prediction_output = model(batch_imgs)
                    prediction_logits = prediction_output['logits']

            predicted_classes = torch.argmax(prediction_logits, dim=1)

            # --- 4. SEQUENTIAL METRIC TRACKING (DRIFT DETECTION) ---
            # We iterate through the batch locally to track exactly *when* drift happens
            for i in range(len(batch_imgs)):
                # Because shuffle=False, total_samples_seen perfectly matches df index
                row_concept = df.iloc[total_samples_seen]['concept']

                if current_concept is None:
                    current_concept = row_concept
                elif row_concept != current_concept:
                    print(f"\n--- DRIFT DETECTED: Transition da Concept {current_concept} a Concept {row_concept} ---")
                    history['drift_points'].append(total_samples_seen)
                    concept_correct = 0
                    concept_number_images = 0
                    current_concept = row_concept

                is_correct = (predicted_classes[i] == batch_labels[i]).item()
                if is_correct:
                    concept_correct += 1
                    global_correct += 1

                concept_number_images += 1
                global_total += 1
                total_samples_seen += 1

                acc_corrente = (concept_correct / concept_number_images) * 100
                history['total_samples_seen'].append(total_samples_seen)
                history['cumulative_accuracy'].append(acc_corrente)

            # --- 5. BIVIO INFERENZA VS TRAINING ---
            if inference_only:
                continue

            # --- 6. TRAINING (ONLINE LEARNING) ---
            model.train()
            if distillator is not None:
                distillator.train()

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

            if ema is not None:
                ema.update(model)

        # --- 7. FINAL METRICS ---
        final_acc = (concept_correct / concept_number_images) * 100 if concept_number_images > 0 else 0
        global_acc = (global_correct / global_total) * 100 if global_total > 0 else 0

        print(f"\n--- Fine Stream | Accuratezza Finale Concept {current_concept}: {final_acc:.2f}% ---")
        print(f"--- Accuratezza Globale (Tutti i Concept): {global_acc:.2f}% ---")

        return final_acc, history