import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from Config import Config
from FmowManager import FmowManager
from TrainingManager import TrainingManager
from FoundationModel import FoundationModel
from FeatureDistillation import LinearDistiller, MLPDistiller
from Student import Student
from StudentViT import StudentVit
from ExperimentRunner import ExperimentRunner


def setup_device():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 55}")
    print(f" DEVICE SETUP: Using {device.type.upper()}")
    if device.type == 'cuda':
        print(f" GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 55}\n")
    return device


def run_fm_pretraining(device, preDF, manager, training_manager, config):
    fm_model = FoundationModel(use_lora=True).to(device)
    print("\n--- STEP 3: Extended Pretraining unsupervised Foundation Model on all data from 2002-2013 ---")

    if os.path.exists(config.fm_weights_path):
        print(f"Found existing Foundation Model weights at {config.fm_weights_path}. Loading...")
        fm_model.load_state_dict(torch.load(config.fm_weights_path, map_location=device))
        fm_model.eval()  # Important: set back to eval mode for feature extraction
    else:
        print("No weights found. Commencing DINO Extended Pre-training...")
        training_manager.pretrain_teacher(
            foundation_model=fm_model,
            df=preDF,  # Use all 140k images, not just top 25!
            manager=manager,
            save_path=config.fm_weights_path,
            epochs=25,
            batch_size=64,  # Depends on your VRAM, adjust if OOM
            accumulation_steps=4
        )
        fm_model.eval()
    return fm_model


def run_student_pretraining(device, preDF_sampled, fm_model, manager, training_manager, class_to_idx, config):
    print("\n--- STEP 6: Offline Pretraining on preDF (2002-2013) ---")

    needs_student_only = any(exp in config.experiments for exp in [1, 3, 4, 8, 10])
    needs_student_proj = any(exp in config.experiments for exp in [2, 5, 6, 7, 9])

    weights_paths = {}

    if needs_student_only:
        weights_paths['student_only'] = config.student_weights_path
        if os.path.exists(config.student_weights_path):
            print(f"Found existing pretrained weights at {config.student_weights_path}. Skipping Student-only pretraining.")
        else:
            print(f"Starting offline pretraining loop for Student only ({config.student_type})...")
            if config.student_type == 'vit':
                student_base = StudentVit(numberOfClasses=config.num_classes, pretrained=True).to(device)
            else:
                student_base = Student(numberOfClasses=config.num_classes, pretrained=True).to(device)
            #TODO Attention we need two different files, to save weights for resnet and ViT    
            student_pretrained, _ = training_manager.pretrain_student(
                student=student_base,
                df=preDF_sampled,
                manager=manager,
                class_to_idx=class_to_idx,
                transform_fn=config.transform_imagenet,
                use_embeddings=False,
                epochs=50,
                batch_size=64,
                lr=1e-4,
                patience=5
            )
            print(f"Saving offline pretraining weights to {config.student_weights_path}...")
            torch.save(student_pretrained.state_dict(), config.student_weights_path)
            del student_base, student_pretrained
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if needs_student_proj:
        weights_paths['student_proj'] = config.student_proj_weights_path
        weights_paths['projector'] = config.proj_weights_path

        if os.path.exists(config.student_proj_weights_path) and os.path.exists(config.proj_weights_path):
            print(f"Found existing pretrained weights at {config.student_proj_weights_path}. Skipping Student+Projector pretraining.")
        else:
            print("\n--- Extracting Embeddings for Pre-training with Projector ---")
            fm_model.eval()
            preDF_embed = manager.get_embeddings(
                df_sample=preDF_sampled,
                model=fm_model,
                transform_fn=config.transform_imagenet,
                batch_size=128,
                save_path="./dino_preDF"
            )

            print(f"Starting offline pretraining loop for Student + Projector ({config.student_type} + {config.projector_type})...")
            if config.student_type == 'vit':
                student_base = StudentVit(numberOfClasses=config.num_classes, pretrained=True).to(device)
                dim_student = student_base.student.num_features
            else:
                student_base = Student(numberOfClasses=config.num_classes, pretrained=True).to(device)
                dim_student = 512
                
            if config.projector_type == 'mlp':
                distillator = MLPDistiller(dimFeatureStudent=dim_student, dimFeatureTeacher=768, hiddenLayerSize=config.mlp_hidden_size).to(device)
            else:
                distillator = LinearDistiller(dimFeatureStudent=dim_student, dimFeatureTeacher=768).to(device)

            student_pretrained, proj_pretrained = training_manager.pretrain_student(
                student=student_base,
                df=preDF_embed,
                manager=manager,
                class_to_idx=class_to_idx,
                transform_fn=config.transform_imagenet,
                use_embeddings=True,
                epochs=50,
                batch_size=64,
                lr=1e-4,
                patience=5,
                projector=distillator,
                lambda_distill=config.distill_weight
            )
            print(f"Saving offline pretraining weights to {config.student_proj_weights_path} and {config.proj_weights_path}...")
            torch.save(student_pretrained.state_dict(), config.student_proj_weights_path)
            torch.save(proj_pretrained.state_dict(), config.proj_weights_path)
            del student_base, student_pretrained, distillator, proj_pretrained
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return weights_paths


def prepare_streaming_data_and_eval(device, postDF_sampled, fm_model, manager, training_manager, class_to_idx, config, top_25_classes,config_id):
    print("\n--- STEP 7a: Extracting Embeddings for Raw Foundation Model ---")
    fm_model_raw = FoundationModel(use_lora=False).to(device)
    fm_model_raw.eval()
    
    postDF_raw_embed = manager.get_embeddings(
        df_sample=postDF_sampled,
        model=fm_model_raw,
        transform_fn=config.transform_imagenet,
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
        transform_fn=config.transform_imagenet,
        batch_size=128,
        save_path="./dino"
    )

    print("\n--- STEP 7c: Offline Linear Probing Comparison ---")
    print("Evaluating Raw Foundation Model features...")
    acc_raw = training_manager.train_linear_probe(postDF_raw_embed, class_to_idx, config.num_classes)
    
    print("Evaluating Extended Foundation Model features...")
    acc_ext = training_manager.train_linear_probe(postDF_ext_embed, class_to_idx, config.num_classes)
    
    print("\n=======================================================")
    print(" LINEAR PROBING RESULTS (Offline on Streaming Data)")
    print("=======================================================")
    print(f" Raw FM Accuracy:      {acc_raw * 100:.2f}%")
    print(f" Extended FM Accuracy: {acc_ext * 100:.2f}%")
    print("=======================================================\n")
    
    # Let manager handle the pandas slicing
    stream_df, test_dict = manager.prepare_streaming_concepts(
        postDF_sampled=postDF_ext_embed,
        top_classes=top_25_classes,
        test_size_per_concept=config.test_size_per_concept,
        config_id=config_id
        )

    return stream_df, test_dict


def save_experiment_results(results_payload):
    print("\n--- STEP 8: Exporting Results for Local Plotting ---")
    output_file = "experiment_results.json"
    with open(output_file, "w") as f:
        json.dump(results_payload, f, indent=4)

    print(f"Data successfully saved to {output_file}.")
    print("Download this file to your local machine to generate the matplotlib charts without server X11 errors.")


def main():
    config = Config.parse_arguments()
    device = setup_device()

    manager = FmowManager(device=device)
    training_manager = TrainingManager(device=device)

    preDF_sampled, postDF_sampled, top_25_classes, class_to_idx, preDF = manager.prepare_data_splits(
        num_classes=config.num_classes,
        pre_samples=config.pre_samples_per_class,
        post_samples=config.post_samples_per_class
    )

    fm_model = run_fm_pretraining(device, preDF, manager, training_manager, config)

    weights_paths = run_student_pretraining(
        device, preDF_sampled, fm_model, manager, training_manager, class_to_idx, config
    )

    results_payload = {}
    
    for conf_id in config.concept_configurations:
        print(f"\n{'='*25} RUNNING CONCEPT CONFIGURATION {conf_id} {'='*25}")
        postDF_sampled_conf, test_dict = prepare_streaming_data_and_eval(
            device, postDF_sampled, fm_model, manager, training_manager, class_to_idx, config, top_25_classes, conf_id
        )
        #TODO Avoid calculating embeddding and linear probing multiple times
        # Cleanup FM after embeddings extraction to free VRAM for the streaming phase if needed
        # We'll just do it after the first config or do it inside prepare_streaming_data_and_eval?
        # Note: fm_model is needed for each configuration's embedding extraction unless we extract once.
        # But prepare_streaming_data_and_eval does extraction. Actually, extraction is independent of concept grouping.
        # So we could extract once, but to keep it simple we just let it run or rely on the disk cache.

        runner = ExperimentRunner(device, config, manager, training_manager)
        conf_results = runner.run_experiments(
            experiments=config.experiments,
            df_sampled=postDF_sampled_conf,
            class_to_idx=class_to_idx,
            weights_paths=weights_paths,
            test_dict=test_dict
        )
        results_payload[f"config_{conf_id}"] = conf_results

    # Cleanup FM
    del fm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    save_experiment_results(results_payload)


if __name__ == "__main__":
    main()