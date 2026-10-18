import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List

from Student import Student
from StudentViT import StudentVit
from EmaTeacher import EmaTeacher
from FeatureDistillation import LinearDistiller, MLPDistiller, RKDDistiller
from FoundationModel import FoundationModel, FoundationModelLinearProbe
from Config import Config
from FmowManager import FmowManager
from TrainingManager import TrainingManager


@dataclass
class ExperimentSetup:
    """
    Configuration specification for a single streaming experiment.

    Attributes:
        description (str): Human-readable name/description of the experimental condition.
        student_weights_key (str): Key identifying which historical pretrained weights to load
                                   ('student_only', 'student_proj', or 'fm_linear_probe').
        projector_weights_key (Optional[str]): Key for projector weights ('projector', 'random', or None).
        inference_only (bool): If True, model runs evaluation on stream without backward pass updates.
        distillator_active (bool): Whether distillation loss is computed during streaming fine-tuning.
        freeze_distillator (bool): If True, keeps the projector frozen during streaming distillation.
        use_ema (bool): Whether to use an EMA Teacher model for streaming predictions.
        optimizer_setup (Callable): Factory function constructing the appropriate optimizer.
        distillation_stop_after (Optional[int]): Sample threshold per drift after which distillation stops.
    """
    description: str
    student_weights_key: str  # e.g., 'student_only' or 'student_proj'
    projector_weights_key: Optional[str] # e.g., 'projector' or None or 'random'
    inference_only: bool
    distillator_active: bool
    freeze_distillator: bool
    use_ema: bool
    optimizer_setup: Callable[[nn.Module, Optional[nn.Module], Config], Optional[optim.Optimizer]]
    distillation_stop_after: Optional[int] = None


def optimizer_student_only(student, distillator, config):
    """Optimizer builder for updating only the student model parameters."""
    return optim.Adam(student.parameters(), lr=config.lr_ft)


def optimizer_student_and_proj(student, distillator, config):
    """
    Optimizer builder for joint fine-tuning:
    Student parameters updated with `lr_ft`, Projector parameters updated with `lr_dist_proj`.
    """
    params = [{'params': student.parameters(), 'lr': config.lr_ft}]
    dist_params = list(distillator.parameters())
    if dist_params:
        params.append({'params': dist_params, 'lr': config.lr_dist_proj})
    return optim.Adam(params)


def optimizer_linear_probe(student, distillator, config):
    """Optimizer builder for updating only the linear probe head on top of the frozen FM."""
    return optim.Adam(student.linear_probe.parameters(), lr=config.lr_ft)


def get_experiment_registry() -> Dict[int, ExperimentSetup]:
    """
    Master registry of all experimental conditions investigating how knowledge distillation
    accelerates adaptation and mitigates catastrophic forgetting across concept drift.

    1) Exp 1: Student finetuned on historic + pure inference on stream (Baseline zero-shot / drift decay)
    2) Exp 2: (Student + Projector) finetuned on historic + pure inference on stream (Distillation warm-up effect)
    3) Exp 3: Student finetuned on historic + student finetuning on stream (Standard online fine-tuning)
    4) Exp 4: Student finetuned on historic + (student + projector) finetuning on stream (Emergent streaming distillation)
    5) Exp 5: (Student + Projector) finetuned on historic + (student + Projector) finetuned on stream (Continuous distillation)
    6) Exp 6: (Student + Projector) finetuned on historic + (student finetuned on stream + projector frozen)
    7) Exp 7: (Student + Projector) finetuned on historic + (student finetuned on stream, no projector)
    8) Exp 8: Continuous distillation stopping after 1000 samples per concept (Resource budget test)
    9) Exp 9: Continuous distillation stopping after 2000 samples per concept (Resource budget test)
    10) Exp 10: Continuous distillation stopping after 4000 samples per concept (Resource budget test)
    11) Exp 11: Continuous distillation stopping after 8000 samples per concept (Resource budget test)
    12) Exp 12: Frozen Extended Foundation Model + Online Linear Probing (Upper-bound feature baseline)
    """
    return {
        1: ExperimentSetup(
            description="Student finetuned on historic + pure inference on stream",
            student_weights_key='student_only',
            projector_weights_key=None,
            inference_only=True,
            distillator_active=False,
            freeze_distillator=False,
            use_ema=False,
            optimizer_setup=lambda s, d, c: None
        ),
        2: ExperimentSetup(
            description="(Student + Projector) finetuned on historic + pure inference on stream",
            student_weights_key='student_proj',
            projector_weights_key=None,
            inference_only=True,
            distillator_active=False,
            freeze_distillator=False,
            use_ema=False,
            optimizer_setup=lambda s, d, c: None
        ),
        3: ExperimentSetup(
            description="Student finetuned on historic + student finetuning on stream",
            student_weights_key='student_only',
            projector_weights_key=None,
            inference_only=False,
            distillator_active=False,
            freeze_distillator=False,
            use_ema=False,
            optimizer_setup=optimizer_student_only
        ),
        4: ExperimentSetup(
            description="Student finetuned on historic + (student + projector) finetuning on stream",
            student_weights_key='student_only',
            projector_weights_key='random',
            inference_only=False,
            distillator_active=True,
            freeze_distillator=False,
            use_ema=False,
            optimizer_setup=optimizer_student_and_proj
        ),
        5: ExperimentSetup(
            description="(Student + Projector) finetuned on historic + (student + Projector) finetuned on stream",
            student_weights_key='student_proj',
            projector_weights_key='projector',
            inference_only=False,
            distillator_active=True,
            freeze_distillator=False,
            use_ema=False,
            optimizer_setup=optimizer_student_and_proj
        ),
        6: ExperimentSetup(
            description="(Student + Projector) finetuned on historic + (student finetuned on stream + projector frozen on stream)",
            student_weights_key='student_proj',
            projector_weights_key='projector',
            inference_only=False,
            distillator_active=True,
            freeze_distillator=True,
            use_ema=False,
            optimizer_setup=optimizer_student_only
        ),
        7: ExperimentSetup(
            description="(Student + Projector) finetuned on historic + (student finetuned on stream, no projector)",
            student_weights_key='student_proj',
            projector_weights_key=None,
            inference_only=False,
            distillator_active=False,
            freeze_distillator=False,
            use_ema=False,
            optimizer_setup=optimizer_student_only
        ),
        8: ExperimentSetup(
            description="(Student + Projector) finetuned on historic + Distillation finetuning on stream (Stops after 1000 images per drift)",
            student_weights_key='student_proj',
            projector_weights_key='projector',
            inference_only=False,
            distillator_active=True,
            freeze_distillator=False,
            use_ema=False,
            optimizer_setup=optimizer_student_and_proj,
            distillation_stop_after=1000
        ),
        9: ExperimentSetup(
            description="(Student + Projector) finetuned on historic + Distillation finetuning on stream (Stops after 2000 images per drift)",
            student_weights_key='student_proj',
            projector_weights_key='projector',
            inference_only=False,
            distillator_active=True,
            freeze_distillator=False,
            use_ema=False,
            optimizer_setup=optimizer_student_and_proj,
            distillation_stop_after=2000
        ),
        10: ExperimentSetup(
            description="(Student + Projector) finetuned on historic + Distillation finetuning on stream (Stops after 4000 images per drift)",
            student_weights_key='student_proj',
            projector_weights_key='projector',
            inference_only=False,
            distillator_active=True,
            freeze_distillator=False,
            use_ema=False,
            optimizer_setup=optimizer_student_and_proj,
            distillation_stop_after=4000
        ),
        11: ExperimentSetup(
            description="(Student + Projector) finetuned on historic + Distillation finetuning on stream (Stops after 8000 images per drift)",
            student_weights_key='student_proj',
            projector_weights_key='projector',
            inference_only=False,
            distillator_active=True,
            freeze_distillator=False,
            use_ema=False,
            optimizer_setup=optimizer_student_and_proj,
            distillation_stop_after=8000
        ),
        12: ExperimentSetup(
            description="Frozen Extended FM + Linear Probing finetuning on stream",
            student_weights_key='fm_linear_probe',
            projector_weights_key=None,
            inference_only=False,
            distillator_active=False,
            freeze_distillator=False,
            use_ema=False,
            optimizer_setup=optimizer_linear_probe,
            distillation_stop_after=None
        ),
    }


class ExperimentRunner:
    """
    Execution engine that instantiates models, loads appropriate checkpoint weights,
    and runs experiments across streaming configurations.
    """
    def __init__(self, device: torch.device, config: Config, manager: FmowManager, training_manager: TrainingManager, fm_model: Optional[nn.Module] = None):
        self.device = device
        self.config = config
        self.manager = manager
        self.training_manager = training_manager
        self.fm_model = fm_model
        self.registry = get_experiment_registry()
        
    def _get_student(self, weights_key: str, weights_paths: Dict[str, str], num_classes: int) -> nn.Module:
        """
        Instantiates student model or linear probe and restores the corresponding checkpoint weights.
        """
        if weights_key == 'fm_linear_probe':
            if self.fm_model is None:
                fm = FoundationModel(use_lora=True).to(self.device)
                if os.path.exists(self.config.fm_weights_path):
                    fm.load_state_dict(torch.load(self.config.fm_weights_path, map_location=self.device))
                fm.eval()
            else:
                fm = self.fm_model
            model = FoundationModelLinearProbe(foundation_model=fm, num_classes=num_classes).to(self.device)
            if weights_key in weights_paths and os.path.exists(weights_paths[weights_key]):
                model.linear_probe.load_state_dict(torch.load(weights_paths[weights_key], map_location=self.device))
            return model

        # Instantiate student model according to student_type ('resnet' or 'vit')
        if self.config.student_type == 'vit':
            s = StudentVit(numberOfClasses=num_classes, pretrained=False).to(self.device)
        else:
            s = Student(numberOfClasses=num_classes, pretrained=False).to(self.device)
            
        if weights_key in weights_paths:
            s.load_state_dict(torch.load(weights_paths[weights_key], map_location=self.device))
        return s

    def _get_distillator(self, weights_key: Optional[str], weights_paths: Dict[str, str], student: nn.Module) -> Optional[nn.Module]:
        """
        Instantiates projector module ('linear', 'mlp', or 'rkd') and restores pretrained weights if requested.
        """
        if weights_key is None:
            return None
            
        # Determine student output dimension
        dim_student = student.student.num_features if self.config.student_type == 'vit' else 512
        
        if self.config.projector_type == 'mlp':
            d = MLPDistiller(dimFeatureStudent=dim_student, dimFeatureTeacher=768, hiddenLayerSize=self.config.mlp_hidden_size).to(self.device)
        elif self.config.projector_type == 'rkd':
            d = RKDDistiller().to(self.device)
        else:
            d = LinearDistiller(dimFeatureStudent=dim_student, dimFeatureTeacher=768).to(self.device)
            
        if weights_key != 'random' and weights_key in weights_paths:
            if self.config.projector_type != 'rkd':
                d.load_state_dict(torch.load(weights_paths[weights_key], map_location=self.device))
        return d

    def run_experiments(self, experiments: List[int], df_sampled, class_to_idx: Dict[str, int], weights_paths: Dict[str, str], test_dict: Dict) -> Dict[str, Any]:
        """
        Executes a sequence of experiments on the provided streaming dataframe partition.

        Args:
            experiments (List[int]): List of experiment IDs to run.
            df_sampled (pd.DataFrame): Streaming dataframe containing ordered concepts.
            class_to_idx (dict): Class name to integer index mapping.
            weights_paths (dict): Paths to pretrained student and projector weights.
            test_dict (dict): Dictionary of held-out test splits per concept for CL evaluation.

        Returns:
            dict: Results payload containing final accuracy, history trajectories, and CL matrices.
        """
        results_payload = {}
        criterion = nn.CrossEntropyLoss()

        for exp_id in experiments:
            if exp_id not in self.registry:
                print(f"Warning: Experiment {exp_id} not found in registry. Skipping.")
                continue

            setup = self.registry[exp_id]
            
            print("\n" + "=" * 55)
            print(f" EXPERIMENT {exp_id}: {setup.description}")
            print("=" * 55)

            student = self._get_student(setup.student_weights_key, weights_paths, self.config.num_classes)
            distillator = self._get_distillator(setup.projector_weights_key, weights_paths, student)
            optimizer = setup.optimizer_setup(student, distillator, self.config)
            
            ema = None
            if setup.use_ema:
                ema = EmaTeacher(student, alpha=self.config.ema_alpha).to(self.device)

            acc, hist, cl = self.training_manager.train_online(
                model=student,
                df=df_sampled,
                criterion=criterion,
                optimizer=optimizer,
                manager=self.manager,
                transform_fn=self.config.transform_imagenet,
                class_to_idx=class_to_idx,
                inference_only=setup.inference_only,
                distillator=distillator if setup.distillator_active else None,
                distill_weight=self.config.distill_weight,
                ema=ema,
                num_epochs_per_batch=self.config.stream_epochs,
                batch_size=self.config.stream_batch_size,
                test_dict=test_dict,
                window_size=self.config.rolling_window_size,
                adaptation_window_size=self.config.adaptation_window_size,
                freeze_distillator=setup.freeze_distillator,
                use_ce_masking=self.config.use_ce_masking,
                distillation_stop_after=setup.distillation_stop_after
            )
            
            results_payload[f"exp_{exp_id}"] = {
                "final_accuracy": acc, 
                "history": hist, 
                "cl_matrix": cl
            }

            # Cleanup GPU memory between experiment runs
            del student
            if distillator:
                del distillator
            if optimizer:
                del optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results_payload

