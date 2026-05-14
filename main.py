import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from FmowManager import FmowManager
from TrainingManager import TrainingManager
from FoundationModel import FoundationModel
from FeatureDistillation import LinearReluDistiller
from Student import Student


def main():
    # --- DEVICE CHECK ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 55}")
    print(f" DEVICE SETUP: Using {device.type.upper()}")
    if device.type == 'cuda':
        print(f" GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 55}\n")

    # Pass the device to the managers
    manager = FmowManager(device=device)
    training_manager = TrainingManager(device=device)

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

    print("\n--- STEP 3: Sample 1500 images per class in preDF (top 25 classes) ---")
    preDF_sampled = manager.sample_dataset(
        preDF_top25,
        category_col='category',
        samples_per_class=2200,
        random_state=42
    )

    print("\n--- STEP 4: Sample 500 images per class in postDF, shuffle, create/order by concepts ---")
    # Filter postDF to include the SAME 25 classes
    postDF_top25 = postDF[postDF['category'].isin(top_25_classes)].copy()

    postDF_sampled = manager.sample_dataset(
        postDF_top25,
        category_col='category',
        samples_per_class=2800,
        random_state=42
    )

    # Setup Transforms (ImageNet standard for ResNet18 and DINOv2)
    transform_imagenet = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    criterion = nn.CrossEntropyLoss()
    num_classes = 25

    print("\n--- STEP 5: Offline Pretraining on preDF (2002-2013) ---")

    # ---------------------------------------------------------
    # PATH TO YOUR PRETRAINED WEIGHTS
    # If this file exists, the script will skip pretraining.
    # ---------------------------------------------------------
    weights_path = "./pretrained_student_weights.pth"

    if os.path.exists(weights_path):
        print(f"Found existing pretrained weights at {weights_path}. Skipping pretraining.")
    else:
        print("Starting offline pretraining loop...")
        # Initialize with ImageNet weights before fine-tuning on preDF
        student_base = Student(numberOfClasses=num_classes, pretrained=True).to(device)

        # Adjust epochs, batch_size, and patience as needed for your server
        student_pretrained, _ = training_manager.pretrain_student(
            student=student_base,
            df=preDF_sampled,
            manager=manager,
            class_to_idx=class_to_idx,
            transform_fn=transform_imagenet,
            use_embeddings=False,
            epochs=100,
            batch_size=64,
            lr=1e-3,
            patience=5
        )

        print(f"Saving offline pretraining weights to {weights_path}...")
        torch.save(student_pretrained.state_dict(), weights_path)

        # Free memory
        del student_base, student_pretrained
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    '''
    print("\n--- STEP 6: Extracting Foundation Model Embeddings for PostDF ---")
    fm_model = FoundationModel().to(device)

    # We need the embeddings explicitly calculated before the stream
    postDF_sampled = manager.get_embeddings(
        df_sample=postDF_sampled,
        model=fm_model,
        transform_fn=transform_imagenet,
        batch_size=128,
        save_path="./dino"
    )
    
    # Free up memory before streaming
    del fm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    '''


    # Shuffle the dataset
    postDF_sampled = postDF_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

    # Mocking a class_to_concept mapping
    dummy_concept_mapping = {cls: (f"Concept_{(i % 5)}") for i, cls in enumerate(top_25_classes)}
    postDF_sampled = manager.create_concepts(postDF_sampled, dummy_concept_mapping)

    # Order by the newly created concepts
    postDF_sampled = postDF_sampled.sort_values(by='concept').reset_index(drop=True)



    print("\n--- STEP 7: STREAMING EXPERIMENTS (2016-2017) ---")

    print("\n=======================================================")
    print(" EXPERIMENT 1: PURE INFERENCE (Pretrained Student)")
    print("=======================================================")
    # Initialize without ImageNet weights, then load our custom preDF weights
    student_inf = Student(numberOfClasses=num_classes, pretrained=False).to(device)
    student_inf.load_state_dict(torch.load(weights_path, map_location=device))

    acc_inf, hist_inf = training_manager.train_online(
        model=student_inf,
        df=postDF_sampled,
        criterion=criterion,
        optimizer=None,
        manager = manager,
        transform_fn=transform_imagenet,
        class_to_idx=class_to_idx,
        inference_only=True,
        num_epochs_per_batch=1,
        batch_size=50
    )

    print("\n=======================================================")
    print(" EXPERIMENT 2: ONLINE FINE-TUNING (Pretrained Student)")
    print("=======================================================")
    student_ft = Student(numberOfClasses=num_classes, pretrained=False).to(device)
    student_ft.load_state_dict(torch.load(weights_path, map_location=device))

    optimizer_ft = optim.Adam(student_ft.parameters(), lr=1e-3)

    acc_ft, hist_ft = training_manager.train_online(
        model=student_ft,
        df=postDF_sampled,
        criterion=criterion,
        optimizer=optimizer_ft,
        manager = manager,
        transform_fn=transform_imagenet,
        class_to_idx=class_to_idx,
        inference_only=False,
        distillator=None,
        num_epochs_per_batch=1,
        batch_size=50
    )


    '''
    print("\n=======================================================")
    print(" EXPERIMENT 3: ONLINE FINE-TUNING + DISTILLATION (Pretrained Student)")
    print("=======================================================")
    student_dist = Student(numberOfClasses=num_classes, pretrained=False).to(device)
    student_dist.load_state_dict(torch.load(weights_path))

    distillator = LinearReluDistiller(dimFeatureStudent=512, dimFeatureTeacher=768).to(device)

    optimizer_dist = optim.Adam([
        {'params': student_dist.parameters(), 'lr': 1e-3},
        {'params': distillator.parameters(), 'lr': 1e-2}
    ])

    acc_dist, hist_dist = training_manager.train_online(
        model=student_dist,
        df=postDF_sampled,
        criterion=criterion,
        optimizer=optimizer_dist,
        get_input_fn=manager.get_input,
        transform_fn=transform_imagenet,
        class_to_idx=class_to_idx,
        inference_only=False,
        distillator=distillator,
        distill_weight=1.0,
        num_epochs_per_batch=1
    )
    '''

    print("\n--- STEP 8: Exporting Results for Local Plotting ---")
    results_payload = {
        "inference": {
            "final_accuracy": acc_inf,
            "history": hist_inf
        },
        "fine_tuning": {
            "final_accuracy": acc_ft,
            "history": hist_ft
        },
    }

    output_file = "experiment_results.json"
    with open(output_file, "w") as f:
        json.dump(results_payload, f, indent=4)

    print(f"Data successfully saved to {output_file}.")
    print("Download this file to your local machine to generate the matplotlib charts without server X11 errors.")

if __name__ == "__main__":
    main()