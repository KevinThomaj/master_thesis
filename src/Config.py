import argparse
from dataclasses import dataclass, field
from typing import List
import torchvision.transforms as T


@dataclass
class Config:
    """
    Central experiment configuration and hyperparameter container.

    Holds settings for:
    - Online streaming parameters (batch size, epochs per batch, evaluation windows).
    - Model architecture choices (ResNet-18 vs ViT-Tiny student, Linear/MLP/RKD projector).
    - Learning rates and loss weighting for student fine-tuning and feature distillation.
    - Dataset parameters (25 classes, 2200 historical samples/class, 2300 streaming samples/class, 100 test samples/concept).
    - Concept configurations and experiment selection (Experiments 1..12).
    """

    # Streaming Hyperparameters
    stream_batch_size: int = 10          # Batch size for online prequential evaluation & training
    stream_epochs: int = 1               # Number of gradient updates per incoming streaming batch
    use_ce_masking: bool = False         # Whether to mask unseen class logits during streaming
    rolling_window_size: int = 1000      # Window size (samples) for rolling accuracy tracking
    adaptation_window_size: int = 500    # Window size (samples) for adaptation speed bar chart metrics
    
    # Model Configurations
    student_type: str = 'resnet'         # Student backbone: 'resnet' (ResNet-18) or 'vit' (ViT-Tiny)
    ema_alpha: float = 0.99              # Smoothing factor alpha for EMA Teacher
    projector_type: str = 'linear'       # Feature distillation projector: 'linear', 'mlp', or 'rkd'
    mlp_hidden_size: int = 1024          # Hidden dimension when using MLP projector
    
    # Learning Rates
    lr_ft: float = 1e-4                  # Learning rate for student model fine-tuning
    lr_dist_proj: float = 1e-3           # Learning rate for feature distillation projector
    
    # Loss Weights
    distill_weight: float = 1.0          # Multiplier lambda for feature distillation loss
    
    # Experiments and Stream Configurations
    experiments: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    concept_configurations: List[int] = field(default_factory=lambda: list(range(1, 31)))
    recurrent_concept: str = None        # Name of concept to repeat at the end of stream (e.g. 'Concept_0')
    
    # Data params matching the thesis specification
    num_classes: int = 25                # Number of selected stable classes
    pre_samples_per_class: int = 2200    # Historical pre-training images per class (2002-2013)
    post_samples_per_class: int = 2300   # Online streaming images per class (2016-2017)
    test_size_per_concept: int = 100     # Evaluation test set held out per concept for CL matrix
    
    @property
    def fm_weights_path(self) -> str:
        """Path to saved Foundation Model (DINOv2 + LoRA) weights."""
        return "./pretrained_dinov2_lora.pth"

    @property
    def student_weights_path(self) -> str:
        """Path to saved Student weights pretrained with cross-entropy only."""
        return f"./pretrained_student_weights_{self.student_type}.pth"

    @property
    def student_proj_weights_path(self) -> str:
        """Path to saved Student weights pretrained with joint cross-entropy and feature distillation."""
        return f"./pretrained_student_proj_weights_{self.student_type}_{self.projector_type}.pth"

    @property
    def proj_weights_path(self) -> str:
        """Path to saved Projector weights from offline pretraining."""
        return f"./pretrained_projector_weights_{self.student_type}_{self.projector_type}.pth"

    # Default Transforms
    @property
    def transform_imagenet(self):
        """Standard ImageNet preprocessing: ToTensor followed by mean and std normalization."""
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def parse_arguments():
        """
        Parses command-line arguments to configure pipeline execution.

        Returns:
            Config: Populated Config instance.
        """
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
        parser.add_argument('--experiments', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                            help='List of experiments to run. Default is all.')
        parser.add_argument('--concept_configurations', nargs='+', type=int, default=None,
                            help='Explicit list of concept configuration IDs to test (overrides --num_configurations).')
        parser.add_argument('--num_configurations', type=int, default=30,
                            help='Run concept configurations 1..N. Default is 30.')
        parser.add_argument('--rolling_window_size', type=int, default=1000,
                            help='Window size for rolling accuracy calculation.')
        parser.add_argument('--adaptation_window_size', type=int, default=500,
                            help='Window size for adaptation speed calculation.')
        parser.add_argument('--student_type', type=str, default='resnet', choices=['resnet', 'vit'],
                            help='Type of student model: resnet or vit.')
        parser.add_argument('--ema_alpha', type=float, default=0.99,
                            help='Alpha for EMA Teacher.')
        parser.add_argument('--projector_type', type=str, default='linear', choices=['linear', 'mlp', 'rkd'],
                            help='Type of distillator projector.')
        parser.add_argument('--mlp_hidden_size', type=int, default=1024,
                            help='Hidden layer size for MLP projector.')
        parser.add_argument('--use_ce_masking', action='store_true',
                            help='Enable cross-entropy logit masking for unseen classes during streaming.')
        parser.add_argument('--recurrent_concept', type=str, default=None,
                            help='The name of a concept to repeat at the end of the stream (e.g., Concept_1).')

        args = parser.parse_args()

        concept_configurations = (args.concept_configurations
                                  if args.concept_configurations is not None
                                  else list(range(1, args.num_configurations + 1)))
        
        return Config(
            stream_batch_size=args.stream_batch_size,
            stream_epochs=args.stream_epochs,
            lr_ft=args.lr_ft,
            lr_dist_proj=args.lr_dist_proj,
            distill_weight=args.distill_weight,
            experiments=args.experiments,
            concept_configurations=concept_configurations,
            rolling_window_size=args.rolling_window_size,
            adaptation_window_size=args.adaptation_window_size,
            student_type=args.student_type,
            ema_alpha=args.ema_alpha,
            projector_type=args.projector_type,
            mlp_hidden_size=args.mlp_hidden_size,
            use_ce_masking=args.use_ce_masking,
            recurrent_concept=args.recurrent_concept
        )

