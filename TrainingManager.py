import copy
import numpy as np
from lightly.data import DINOCollateFunction
from lightly.loss import DINOLoss
from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule
from torch.cuda.amp import GradScaler, autocast
import os
import collections

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
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
        # --- INIEZIONE PER ARCHITETTURA ADA ---
        #print("Compilazione del modello per ottimizzazione Tensor Cores (richiederà qualche minuto iniziale)...")
        #dino_wrapper = torch.compile(dino_wrapper)

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
            momentum_val = cosine_schedule(epoch, epochs, 0.996, 1.0)

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
                    loss = criterion(teacher_out, student_out, epoch=epoch) / accumulation_steps

                # Backward pass
                loss.backward()
                dino_wrapper.student_head.cancel_last_layer_gradients(current_epoch=epoch)
                total_loss += loss.item() * accumulation_steps

                # Gradient Accumulation & EMA Update
                if ((step + 1) % accumulation_steps == 0) or ((step + 1) == len(data_loader)):
                    optimizer.step()
                    optimizer.zero_grad()

                    # EMA Update for Teacher
                    # Momentum usually scales from 0.996 to 1.0 during training
                    update_momentum(
                        dino_wrapper.student_backbone, dino_wrapper.teacher_backbone, m=momentum_val
                    )
                    update_momentum(
                        dino_wrapper.student_head, dino_wrapper.teacher_head, m=momentum_val
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
                     inference_only=False,
                     test_dict=None,
                     window_size=200,
                     freeze_distillator=False):
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

        test_loaders = {}
        if test_dict is not None:
            for concept, t_df in test_dict.items():
                t_dataset = FmowTorchDataset(
                    df=t_df,
                    manager=manager,
                    class_to_idx=class_to_idx,
                    transform=transform_fn,
                    use_embeddings=use_emb
                )
                test_loaders[concept] = DataLoader(t_dataset, batch_size=batch_size, shuffle=False)

        model = model.to(self.device)

        history = {
            'total_samples_seen': [],
            'cumulative_accuracy': [],
            'drift_points': [],
            'rolling_accuracy': []
        }
        
        rolling_window = collections.deque(maxlen=window_size)
        cl_matrix = []
        
        def evaluate_test_sets(current_concept):
            print(f"\n--- Evaluating CL Matrix for concept: {current_concept} ---")
            if ema is not None:
                eval_model = ema
            else:
                eval_model = model
                eval_model.eval()
                
            evaluations = {}
            with torch.no_grad():
                for test_concept, t_loader in test_loaders.items():
                    concept_correct = 0
                    concept_total = 0
                    for test_batch in t_loader:
                        if use_emb:
                            t_imgs, _, t_labels = test_batch
                        else:
                            t_imgs, t_labels = test_batch
                            
                        t_imgs = t_imgs.to(self.device)
                        t_labels = t_labels.to(self.device)
                        
                        if ema is not None:
                            t_logits = eval_model.predict(t_imgs)
                        else:
                            t_outputs = eval_model(t_imgs)
                            t_logits = t_outputs['logits']
                            
                        t_preds = torch.argmax(t_logits, dim=1)
                        concept_correct += torch.sum(t_preds == t_labels).item()
                        concept_total += t_labels.size(0)
                        
                    eval_acc = (concept_correct / concept_total) * 100 if concept_total > 0 else 0
                    evaluations[test_concept] = eval_acc
                    
            return {
                'train_concept': current_concept,
                'evaluations': evaluations
            }

        # Counters
        global_correct = 0
        global_total = 0
        concept_correct = 0
        concept_number_images = 0
        current_concept = None
        total_samples_seen = 0
        active_classes = None

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
                    #active class in current concept
                    active_class_list = [class_to_idx[c] for c in df[df['concept'] == current_concept]['category'].unique() if c in class_to_idx]
                    active_classes = torch.tensor(active_class_list, dtype=torch.long, device=self.device)
                elif row_concept != current_concept:
                    print(f"\n--- DRIFT DETECTED: Transition da Concept {current_concept} a Concept {row_concept} ---")
                    history['drift_points'].append(total_samples_seen)
                    
                    if test_loaders:
                        cl_matrix.append(evaluate_test_sets(current_concept))
                        
                    concept_correct = 0
                    concept_number_images = 0
                    current_concept = row_concept
                    active_class_list = [class_to_idx[c] for c in df[df['concept'] == current_concept]['category'].unique() if c in class_to_idx]
                    active_classes = torch.tensor(active_class_list, dtype=torch.long, device=self.device)

                is_correct = (predicted_classes[i] == batch_labels[i]).item()
                if is_correct:
                    concept_correct += 1
                    global_correct += 1
                    
                rolling_window.append(is_correct)
                rolling_acc = (sum(rolling_window) / len(rolling_window)) * 100

                concept_number_images += 1
                global_total += 1
                total_samples_seen += 1

                acc_corrente = (concept_correct / concept_number_images) * 100
                history['total_samples_seen'].append(total_samples_seen)
                history['cumulative_accuracy'].append(acc_corrente)
                history['rolling_accuracy'].append(rolling_acc)

            # --- 5. BIVIO INFERENZA VS TRAINING ---
            if inference_only:
                continue

            # --- 6. TRAINING (ONLINE LEARNING) ---
            model.train()
            if distillator is not None:
                if freeze_distillator:
                    distillator.eval()
                else:
                    distillator.train()

            for epoch in range(num_epochs_per_batch):
                optimizer.zero_grad()

                student_output = model(batch_imgs)
                logits = student_output['logits']

                # --- LOGIT MASKING --- (to prevent forgetting in classification layer)
                if active_classes is not None:
                    mask = torch.ones_like(logits, dtype=torch.bool)
                    mask[:, active_classes] = False
                    logits[mask] = -float('inf')

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

        if test_loaders and current_concept is not None:
            cl_matrix.append(evaluate_test_sets(current_concept))

        # --- 7. FINAL METRICS ---
        final_acc = (concept_correct / concept_number_images) * 100 if concept_number_images > 0 else 0
        global_acc = (global_correct / global_total) * 100 if global_total > 0 else 0

        print(f"\n--- Fine Stream | Accuratezza Finale Concept {current_concept}: {final_acc:.2f}% ---")
        print(f"--- Accuratezza Globale (Tutti i Concept): {global_acc:.2f}% ---")

        return final_acc, history, cl_matrix

    def train_linear_probe(self, df_embeddings, class_to_idx, num_classes, epochs=1, lr=1e-3):


        print("\n--- Starting Offline Linear Probing ---")
        
        train_df, test_df = train_test_split(
            df_embeddings, test_size=0.2, stratify=df_embeddings['category'], random_state=42
        )
        
        # Prepare Tensors
        X_train = torch.tensor(np.stack(train_df['embedding'].values), dtype=torch.float32)
        y_train = torch.tensor(train_df['category'].map(class_to_idx).values, dtype=torch.long)
        
        X_test = torch.tensor(np.stack(test_df['embedding'].values), dtype=torch.float32)
        y_test = torch.tensor(test_df['category'].map(class_to_idx).values, dtype=torch.long)
        
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        embed_dim = X_train.shape[1]
        
        classifier = nn.Linear(embed_dim, num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=lr)
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            classifier.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = classifier(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            classifier.eval()
            correct = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = classifier(batch_x)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == batch_y).sum().item()
            
            test_acc = correct / len(test_dataset)
            if test_acc > best_acc:
                best_acc = test_acc
                
        print(f"Linear Probing Completed. Best Test Accuracy: {best_acc * 100:.2f}%")
        return best_acc