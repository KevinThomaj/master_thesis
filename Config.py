import argparse
from dataclasses import dataclass, field
from typing import List
import torchvision.transforms as T

@dataclass
class Config:
    # Streaming Hyperparameters
    stream_batch_size: int = 50
    stream_epochs: int = 1
    use_ce_masking: bool = False
    
    # Model Configurations
    student_type: str = 'resnet'  # 'resnet' or 'vit'
    ema_alpha: float = 0.99
    projector_type: str = 'linear'  # 'linear' or 'mlp'
    mlp_hidden_size: int = 1024
    
    # Learning Rates
    lr_ft: float = 1e-3
    lr_dist_proj: float = 1e-3
    
    # Loss Weights
    distill_weight: float = 1.0
    
    # Experiments
    experiments: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])
    concept_configurations: List[int] = field(default_factory=lambda: [1, 2, 3])
    
    # Data params
    num_classes: int = 25
    pre_samples_per_class: int = 2200
    post_samples_per_class: int = 2300
    test_size_per_concept: int = 100
    
    # Paths
    fm_weights_path: str = "./pretrained_dinov2_lora.pth"
    student_weights_path: str = "./pretrained_student_weights.pth"
    student_proj_weights_path: str = "./pretrained_student_proj_weights.pth"
    proj_weights_path: str = "./pretrained_projector_weights.pth"

    # Default Transforms
    @property
    def transform_imagenet(self):
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Fmow Streaming Experiments Pipeline")
        parser.add_argument('--stream_batch_size', type=int, default=10,
                            help='Batch size for the online streaming phase.')
        parser.add_argument('--stream_epochs', type=int, default=1,
                            help='Number of training epochs per incoming batch.')
        parser.add_argument('--lr_ft', type=float, default=1e-4,
                            help='Learning rate for standard online fine-tuning.')
        parser.add_argument('--lr_dist_proj', type=float, default=1e-3,
                            help='Learning rate for the distillator projector.')
        parser.add_argument('--distill_weight', type=float, default=1.0,
                            help='Weight lambda for the distillation loss component.')
        parser.add_argument('--experiments', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            help='List of experiments to run. Default is all.')
        parser.add_argument('--concept_configurations', nargs='+', type=int, default=[1, 2, 3],
                            help='List of concept configurations to test. Default is 1, 2, 3.')
        parser.add_argument('--student_type', type=str, default='resnet', choices=['resnet', 'vit'],
                            help='Type of student model: resnet or vit.')
        parser.add_argument('--ema_alpha', type=float, default=0.99,
                            help='Alpha for EMA Teacher.')
        parser.add_argument('--projector_type', type=str, default='linear', choices=['linear', 'mlp'],
                            help='Type of distillator projector.')
        parser.add_argument('--mlp_hidden_size', type=int, default=1024,
                            help='Hidden layer size for MLP projector.')
        parser.add_argument('--use_ce_masking', action='store_true',
                            help='Enable cross-entropy logit masking for unseen classes during streaming.')

        args = parser.parse_args()
        
        return Config(
            stream_batch_size=args.stream_batch_size,
            stream_epochs=args.stream_epochs,
            lr_ft=args.lr_ft,
            lr_dist_proj=args.lr_dist_proj,
            distill_weight=args.distill_weight,
            experiments=args.experiments,
            concept_configurations=args.concept_configurations,
            student_type=args.student_type,
            ema_alpha=args.ema_alpha,
            projector_type=args.projector_type,
            mlp_hidden_size=args.mlp_hidden_size,
            use_ce_masking=args.use_ce_masking
        )
