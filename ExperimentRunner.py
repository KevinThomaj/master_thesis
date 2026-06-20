import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List

from Student import Student
from StudentViT import StudentVit
from EmaTeacher import EmaTeacher
from FeatureDistillation import LinearDistiller, MLPDistiller
from Config import Config
from FmowManager import FmowManager
from TrainingManager import TrainingManager

@dataclass
class ExperimentSetup:
    description: str
    student_weights_key: str  # e.g., 'student_only' or 'student_proj'
    projector_weights_key: Optional[str] # e.g., 'projector' or None or 'random'
    inference_only: bool
    distillator_active: bool
    freeze_distillator: bool
    use_ema: bool
    optimizer_setup: Callable[[nn.Module, Optional[nn.Module], Config], Optional[optim.Optimizer]]

def optimizer_student_only(student, distillator, config):
    return optim.Adam(student.parameters(), lr=config.lr_ft)

def optimizer_student_and_proj(student, distillator, config):
    return optim.Adam([
        {'params': student.parameters(), 'lr': config.lr_ft},
        {'params': distillator.parameters(), 'lr': config.lr_dist_proj}
    ])

def get_experiment_registry() -> Dict[int, ExperimentSetup]:
    """
    Registry of all available experiments.
    To add a new experiment, simply add a new entry to this dictionary with a unique ID.
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
            description="Student finetuned on historic + EMA Teacher finetuning on stream",
            student_weights_key='student_only',
            projector_weights_key=None,
            inference_only=False,
            distillator_active=False,
            freeze_distillator=False,
            use_ema=True,
            optimizer_setup=optimizer_student_only
        ),
        9: ExperimentSetup(
            description="(Student + Projector) finetuned on historic + Distillation finetuning on stream (Depends on Config Projector Type)",
            student_weights_key='student_proj',
            projector_weights_key='projector',
            inference_only=False,
            distillator_active=True,
            freeze_distillator=False,
            use_ema=False,
            optimizer_setup=optimizer_student_and_proj
        ),
        10: ExperimentSetup(
            description="Student finetuned on historic + Student finetuning on stream (Depends on Config Student Type)",
            student_weights_key='student_only',
            projector_weights_key=None,
            inference_only=False,
            distillator_active=False,
            freeze_distillator=False,
            use_ema=False,
            optimizer_setup=optimizer_student_only
        ),
    }

class ExperimentRunner:
    def __init__(self, device: torch.device, config: Config, manager: FmowManager, training_manager: TrainingManager):
        self.device = device
        self.config = config
        self.manager = manager
        self.training_manager = training_manager
        self.registry = get_experiment_registry()
        
    def _get_student(self, weights_key: str, weights_paths: Dict[str, str], num_classes: int) -> nn.Module:
        if self.config.student_type == 'vit':
            s = StudentVit(numberOfClasses=num_classes, pretrained=False).to(self.device)
        else:
            s = Student(numberOfClasses=num_classes, pretrained=False).to(self.device)
            
        if weights_key in weights_paths:
            s.load_state_dict(torch.load(weights_paths[weights_key], map_location=self.device))
        return s

    def _get_distillator(self, weights_key: Optional[str], weights_paths: Dict[str, str], student: nn.Module) -> Optional[nn.Module]:
        if weights_key is None:
            return None
            
        # Determine student output dimension
        dim_student = student.student.num_features if self.config.student_type == 'vit' else 512
        
        if self.config.projector_type == 'mlp':
            d = MLPDistiller(dimFeatureStudent=dim_student, dimFeatureTeacher=768, hiddenLayerSize=self.config.mlp_hidden_size).to(self.device)
        else:
            d = LinearDistiller(dimFeatureStudent=dim_student, dimFeatureTeacher=768).to(self.device)
            
        if weights_key != 'random' and weights_key in weights_paths:
            d.load_state_dict(torch.load(weights_paths[weights_key], map_location=self.device))
        return d

    def run_experiments(self, experiments: List[int], df_sampled, class_to_idx: Dict[str, int], weights_paths: Dict[str, str], test_dict: Dict) -> Dict[str, Any]:
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
                freeze_distillator=setup.freeze_distillator,
                use_ce_masking=self.config.use_ce_masking
            )
            
            results_payload[f"exp_{exp_id}"] = {
                "final_accuracy": acc, 
                "history": hist, 
                "cl_matrix": cl
            }

            # Cleanup
            del student
            if distillator:
                del distillator
            if optimizer:
                del optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results_payload
