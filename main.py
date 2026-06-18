import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import argparse
import pandas as pd

from FmowManager import FmowManager
from TrainingManager import TrainingManager
from FoundationModel import FoundationModel
from FeatureDistillation import LinearDistiller
from Student import Student
from sklearn.model_selection import train_test_split


def parse_arguments():
    # --- CLI ARGUMENTS SETUP ---
    parser = argparse.ArgumentParser(description="Fmow Streaming Experiments Pipeline")

    # Streaming Hyperparameters
    parser.add_argument('--stream_batch_size', type=int, default=50,
                        help='Batch size for the online streaming phase.')
    parser.add_argument('--stream_epochs', type=int, default=1,
                        help='Number of training epochs per incoming batch.')

    # Learning Rates
    parser.add_argument('--lr_ft', type=float, default=1e-2,
                        help='Learning rate for the standard online fine-tuning experiment.')
    parser.add_argument('--lr_dist_student', type=float, default=1e-2,
                        help='Learning rate for the student model during distillation.')
    parser.add_argument('--lr_dist_proj', type=float, default=1e-3,
                        help='Learning rate for the distillator projector.')

    # Loss Weights
    parser.add_argument('--distill_weight', type=float, default=1.0,
                        help='Weight lambda for the distillation loss component.')
                        
    # Experiments
    parser.add_argument('--experiments', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6, 7],
                        help='List of experiments to run (1-7). Default is all.')

    return parser.parse_args()


def setup_device():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # --- DEVICE CHECK ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 55}")
    print(f" DEVICE SETUP: Using {device.type.upper()}")
    if device.type == 'cuda':
        print(f" GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 55}\n")
    return device


def prepare_and_sample_data(manager):
    print("\n--- STEP 1: Total Dataset ---")
    total_df = manager.get_dataset()

    print("\n--- STEP 2: Divide into preDF (2002-2013) and postDF (2016-2017) ---")
    preDF, postDF = manager.divide(total_df)

    # Find the 25 most popular classes in preDF
    top_25_classes = preDF['category'].value_counts().nlargest(25).index.tolist()
    print(f"Top 25 classes identified: {top_25_classes}")

    # Filter preDF to only include these top 25 classes
    preDF_top25 = preDF[preDF['category'].isin(top_25_classes)].copy()

    # Create the mapping required for the TorchDataset
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(top_25_classes)}

    print("\n--- STEP 4: Sample 2200 images per class in preDF (top 25 classes) ---")
    preDF_sampled = manager.sample_dataset(
        preDF_top25,
        category_col='category',
        samples_per_class=2200,
        random_state=42
    )

    print("\n--- STEP 5: Sample 2300 images per class in postDF, shuffle, create/order by concepts ---")
    # Filter postDF to include the SAME 25 classes
    postDF_top25 = postDF[postDF['category'].isin(top_25_classes)].copy()

    postDF_sampled = manager.sample_dataset(
        postDF_top25,
        category_col='category',
        samples_per_class=2300,
        random_state=42
    )

    return preDF_sampled, postDF_sampled, top_25_classes, class_to_idx, preDF


def run_fm_pretraining(device, preDF, manager, training_manager):
    fm_model = FoundationModel(use_lora=True).to(device)
    fm_weights_path = "./pretrained_dinov2_lora.pth"
    print("\n--- STEP 3: Extended Pretraining unsupervised Foundation Model on all data from 2002-2013 ---")

    if os.path.exists(fm_weights_path):
        print(f"Found existing Foundation Model weights at {fm_weights_path}. Loading...")
        fm_model.load_state_dict(torch.load(fm_weights_path, map_location=device))
        fm_model.eval()  # Important: set back to eval mode for feature extraction
    else:
        print("No weights found. Commencing DINO Extended Pre-training...")
        training_manager.pretrain_teacher(
            foundation_model=fm_model,
            df=preDF,  # Use all 140k images, not just top 25!
            manager=manager,
            save_path=fm_weights_path,
            epochs=25,
            batch_size=64,  # Depends on your VRAM, adjust if OOM
            accumulation_steps=4
        )
        fm_model.eval()
    return fm_model


def run_student_pretraining(device, preDF_sampled, fm_model, manager, training_manager, class_to_idx, transform_imagenet, num_classes, args):
    print("\n--- STEP 6: Offline Pretraining on preDF (2002-2013) ---")

    experiments = args.experiments
    needs_student_only = any(exp in experiments for exp in [1, 3, 4])
    needs_student_proj = any(exp in experiments for exp in [2, 5, 6, 7])

    weights_paths = {}

    if needs_student_only:
        weights_path_student = "./pretrained_student_weights.pth"
        weights_paths['student_only'] = weights_path_student
        if os.path.exists(weights_path_student):
            print(f"Found existing pretrained weights at {weights_path_student}. Skipping Student-only pretraining.")
        else:
            print("Starting offline pretraining loop for Student only...")
            student_base = Student(numberOfClasses=num_classes, pretrained=True).to(device)
            student_pretrained, _ = training_manager.pretrain_student(
                student=student_base,
                df=preDF_sampled,
                manager=manager,
                class_to_idx=class_to_idx,
                transform_fn=transform_imagenet,
                use_embeddings=False,
                epochs=50,
                batch_size=64,
                lr=1e-4,
                patience=5
            )
            print(f"Saving offline pretraining weights to {weights_path_student}...")
            torch.save(student_pretrained.state_dict(), weights_path_student)
            del student_base, student_pretrained
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if needs_student_proj:
        weights_path_student_proj = "./pretrained_student_proj_weights.pth"
        weights_path_proj = "./pretrained_projector_weights.pth"
        weights_paths['student_proj'] = weights_path_student_proj
        weights_paths['projector'] = weights_path_proj

        if os.path.exists(weights_path_student_proj) and os.path.exists(weights_path_proj):
            print(f"Found existing pretrained weights at {weights_path_student_proj}. Skipping Student+Projector pretraining.")
        else:
            print("\n--- Extracting Embeddings for Pre-training with Projector ---")
            fm_model.eval()
            preDF_embed = manager.get_embeddings(
                df_sample=preDF_sampled,
                model=fm_model,
                transform_fn=transform_imagenet,
                batch_size=128,
                save_path="./dino_preDF"
            )

            print("Starting offline pretraining loop for Student + Projector...")
            student_base = Student(numberOfClasses=num_classes, pretrained=True).to(device)
            distillator = LinearDistiller(dimFeatureStudent=512, dimFeatureTeacher=768).to(device)

            student_pretrained, proj_pretrained = training_manager.pretrain_student(
                student=student_base,
                df=preDF_embed,
                manager=manager,
                class_to_idx=class_to_idx,
                transform_fn=transform_imagenet,
                use_embeddings=True,
                epochs=50,
                batch_size=64,
                lr=1e-4,
                patience=5,
                projector=distillator,
                lambda_distill=args.distill_weight
            )
            print(f"Saving offline pretraining weights to {weights_path_student_proj} and {weights_path_proj}...")
            torch.save(student_pretrained.state_dict(), weights_path_student_proj)
            torch.save(proj_pretrained.state_dict(), weights_path_proj)
            del student_base, student_pretrained, distillator, proj_pretrained
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return weights_paths


def prepare_streaming_data(device, postDF_sampled, fm_model, manager, training_manager, class_to_idx, transform_imagenet, num_classes, top_25_classes):
    print("\n--- STEP 7a: Extracting Embeddings for Raw Foundation Model ---")
    fm_model_raw = FoundationModel(use_lora=False).to(device)
    fm_model_raw.eval()
    
    postDF_raw_embed = manager.get_embeddings(
        df_sample=postDF_sampled,
        model=fm_model_raw,
        transform_fn=transform_imagenet,
        batch_size=128,
        save_path="./dino_noExt"
    )
    
    del fm_model_raw
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n--- STEP 7b: Extracting Embeddings for Extended Foundation Model ---")
    postDF_ext_embed = manager.get_embeddings(
        df_sample=postDF_sampled,
        model=fm_model,
        transform_fn=transform_imagenet,
        batch_size=128,
        save_path="./dino"
    )
    
    del fm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n--- STEP 7c: Offline Linear Probing Comparison ---")
    print("Evaluating Raw Foundation Model features...")
    acc_raw = training_manager.train_linear_probe(postDF_raw_embed, class_to_idx, num_classes)
    
    print("Evaluating Extended Foundation Model features...")
    acc_ext = training_manager.train_linear_probe(postDF_ext_embed, class_to_idx, num_classes)
    
    print("\n=======================================================")
    print(" LINEAR PROBING RESULTS (Offline on Streaming Data)")
    print("=======================================================")
    print(f" Raw FM Accuracy:      {acc_raw * 100:.2f}%")
    print(f" Extended FM Accuracy: {acc_ext * 100:.2f}%")
    print("=======================================================\n")
    
    # Shuffle and prepare the extended embeddings for the streaming experiments
    postDF_sampled_final = postDF_ext_embed.sample(frac=1, random_state=42).reset_index(drop=True)

    # Mocking a class_to_concept mapping
    dummy_concept_mapping = {cls: (f"Concept_{(i % 5)}") for i, cls in enumerate(top_25_classes)}
    postDF_sampled_final = manager.create_concepts(postDF_sampled_final, dummy_concept_mapping)

    # Order by the newly created concepts
    postDF_sampled_final = postDF_sampled_final.sort_values(by='concept').reset_index(drop=True)

    # Split into stream_df and test_dict
    stream_parts = []
    test_dict = {}
    
    # Group by concept and split
    for concept, group in postDF_sampled_final.groupby('concept', sort=False):
        # Taking 20 images per class = 100 images per concept
        test_size = 100
        
        #balanced across classes inside the concept
        stream_part, test_part = train_test_split(
            group, 
            test_size=test_size, 
            stratify=group['category'], 
            random_state=42
        )
            
        stream_parts.append(stream_part)
        if len(test_part) > 0:
            test_dict[concept] = test_part
            
    # Re-concatenate the stream parts
    stream_df = pd.concat(stream_parts).reset_index(drop=True)

    return stream_df, test_dict


def run_streaming_experiments(device, postDF_sampled, manager, training_manager, class_to_idx, transform_imagenet, args, num_classes, weights_paths, test_dict):
    print("\n--- STEP 8: STREAMING EXPERIMENTS (2016-2017) ---")

    criterion = nn.CrossEntropyLoss()
    experiments = args.experiments
    results_payload = {}

    def get_student(weights_key):
        s = Student(numberOfClasses=num_classes, pretrained=False).to(device)
        if weights_key in weights_paths:
            s.load_state_dict(torch.load(weights_paths[weights_key], map_location=device))
        return s

    def get_distillator(weights_key=None):
        d = LinearDistiller(dimFeatureStudent=512, dimFeatureTeacher=768).to(device)
        if weights_key and weights_key in weights_paths:
            d.load_state_dict(torch.load(weights_paths[weights_key], map_location=device))
        return d

    if 1 in experiments:
        print("\n=======================================================")
        print(" EXPERIMENT 1: Student finetuned on historic + pure inference on stream")
        print("=======================================================")
        student = get_student('student_only')
        acc, hist, cl = training_manager.train_online(
            model=student, df=postDF_sampled, criterion=criterion, optimizer=None,
            manager=manager, transform_fn=transform_imagenet, class_to_idx=class_to_idx,
            inference_only=True, num_epochs_per_batch=args.stream_epochs,
            batch_size=args.stream_batch_size, test_dict=test_dict
        )
        results_payload["exp_1"] = {"final_accuracy": acc, "history": hist, "cl_matrix": cl}

    if 2 in experiments:
        print("\n=======================================================")
        print(" EXPERIMENT 2: (Student + Projector) finetuned on historic + pure inference on stream")
        print("=======================================================")
        student = get_student('student_proj')
        acc, hist, cl = training_manager.train_online(
            model=student, df=postDF_sampled, criterion=criterion, optimizer=None,
            manager=manager, transform_fn=transform_imagenet, class_to_idx=class_to_idx,
            inference_only=True, num_epochs_per_batch=args.stream_epochs,
            batch_size=args.stream_batch_size, test_dict=test_dict
        )
        results_payload["exp_2"] = {"final_accuracy": acc, "history": hist, "cl_matrix": cl}

    if 3 in experiments:
        print("\n=======================================================")
        print(" EXPERIMENT 3: Student finetuned on historic + student finetuning on stream")
        print("=======================================================")
        student = get_student('student_only')
        optimizer = optim.Adam(student.parameters(), lr=args.lr_ft)
        acc, hist, cl = training_manager.train_online(
            model=student, df=postDF_sampled, criterion=criterion, optimizer=optimizer,
            manager=manager, transform_fn=transform_imagenet, class_to_idx=class_to_idx,
            inference_only=False, distillator=None,
            num_epochs_per_batch=args.stream_epochs, batch_size=args.stream_batch_size, test_dict=test_dict
        )
        results_payload["exp_3"] = {"final_accuracy": acc, "history": hist, "cl_matrix": cl}

    if 4 in experiments:
        print("\n=======================================================")
        print(" EXPERIMENT 4: Student finetuned on historic + (student + projector) finetuning on stream")
        print("=======================================================")
        student = get_student('student_only')
        distillator = get_distillator() # Random init
        optimizer = optim.Adam([
            {'params': student.parameters(), 'lr': args.lr_dist_student},
            {'params': distillator.parameters(), 'lr': args.lr_dist_proj}
        ])
        acc, hist, cl = training_manager.train_online(
            model=student, df=postDF_sampled, criterion=criterion, optimizer=optimizer,
            manager=manager, transform_fn=transform_imagenet, class_to_idx=class_to_idx,
            inference_only=False, distillator=distillator, distill_weight=args.distill_weight,
            num_epochs_per_batch=args.stream_epochs, batch_size=args.stream_batch_size, test_dict=test_dict
        )
        results_payload["exp_4"] = {"final_accuracy": acc, "history": hist, "cl_matrix": cl}

    if 5 in experiments:
        print("\n=======================================================")
        print(" EXPERIMENT 5: (Student + Projector) finetuned on historic + (student + Projector) finetuned on stream")
        print("=======================================================")
        student = get_student('student_proj')
        distillator = get_distillator('projector')
        optimizer = optim.Adam([
            {'params': student.parameters(), 'lr': args.lr_dist_student},
            {'params': distillator.parameters(), 'lr': args.lr_dist_proj}
        ])
        acc, hist, cl = training_manager.train_online(
            model=student, df=postDF_sampled, criterion=criterion, optimizer=optimizer,
            manager=manager, transform_fn=transform_imagenet, class_to_idx=class_to_idx,
            inference_only=False, distillator=distillator, distill_weight=args.distill_weight,
            num_epochs_per_batch=args.stream_epochs, batch_size=args.stream_batch_size, test_dict=test_dict
        )
        results_payload["exp_5"] = {"final_accuracy": acc, "history": hist, "cl_matrix": cl}

    if 6 in experiments:
        print("\n=======================================================")
        print(" EXPERIMENT 6: (Student + Projector) finetuned on historic + (student finetuned on stream + projector frozen on stream)")
        print("=======================================================")
        student = get_student('student_proj')
        distillator = get_distillator('projector')
        optimizer = optim.Adam(student.parameters(), lr=args.lr_dist_student)
        acc, hist, cl = training_manager.train_online(
            model=student, df=postDF_sampled, criterion=criterion, optimizer=optimizer,
            manager=manager, transform_fn=transform_imagenet, class_to_idx=class_to_idx,
            inference_only=False, distillator=distillator, distill_weight=args.distill_weight,
            num_epochs_per_batch=args.stream_epochs, batch_size=args.stream_batch_size, test_dict=test_dict,
            freeze_distillator=True
        )
        results_payload["exp_6"] = {"final_accuracy": acc, "history": hist, "cl_matrix": cl}

    if 7 in experiments:
        print("\n=======================================================")
        print(" EXPERIMENT 7: (Student + Projector) finetuned on historic + (student finetuned on stream, no projector)")
        print("=======================================================")
        student = get_student('student_proj')
        optimizer = optim.Adam(student.parameters(), lr=args.lr_ft)
        acc, hist, cl = training_manager.train_online(
            model=student, df=postDF_sampled, criterion=criterion, optimizer=optimizer,
            manager=manager, transform_fn=transform_imagenet, class_to_idx=class_to_idx,
            inference_only=False, distillator=None,
            num_epochs_per_batch=args.stream_epochs, batch_size=args.stream_batch_size, test_dict=test_dict
        )
        results_payload["exp_7"] = {"final_accuracy": acc, "history": hist, "cl_matrix": cl}

    return results_payload


def save_experiment_results(results_payload):
    print("\n--- STEP 8: Exporting Results for Local Plotting ---")
    output_file = "experiment_results.json"
    with open(output_file, "w") as f:
        json.dump(results_payload, f, indent=4)

    print(f"Data successfully saved to {output_file}.")
    print("Download this file to your local machine to generate the matplotlib charts without server X11 errors.")


def main():
    args = parse_arguments()
    device = setup_device()

    manager = FmowManager(device=device)
    training_manager = TrainingManager(device=device)

    preDF_sampled, postDF_sampled, top_25_classes, class_to_idx, preDF = prepare_and_sample_data(manager)

    fm_model = run_fm_pretraining(device, preDF, manager, training_manager)

    # Setup Transforms (ImageNet standard for ResNet18 and DINOv2)
    transform_imagenet = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    num_classes = 25

    weights_paths = run_student_pretraining(
        device, preDF_sampled, fm_model, manager, training_manager, class_to_idx, transform_imagenet, num_classes, args
    )

    postDF_sampled, test_dict = prepare_streaming_data(
        device, postDF_sampled, fm_model, manager, training_manager, class_to_idx, transform_imagenet, num_classes, top_25_classes
    )

    results_payload = run_streaming_experiments(
        device, postDF_sampled, manager, training_manager, class_to_idx, transform_imagenet, args, num_classes, weights_paths, test_dict
    )

    save_experiment_results(results_payload)


if __name__ == "__main__":
    main()