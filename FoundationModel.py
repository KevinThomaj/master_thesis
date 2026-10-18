import torch
import torch.nn as nn
import timm
import copy
from peft import LoraConfig, get_peft_model
from lightly.models.modules import DINOProjectionHead
from LinearProbe import LinearProbe


class FoundationModel(nn.Module):
    """
    Ground-based Earth Observation Foundation Model using Meta's DINOv2 ViT-Base backbone.

    Adapted via Low-Rank Adaptation (LoRA) targeting attention projection matrices ('qkv' and 'proj')
    to adapt to historical satellite imagery distribution without full fine-tuning.

    Args:
        use_lora (bool): Whether to inject trainable LoRA adapters or keep the entire backbone frozen.
    """
    def __init__(self, use_lora=True):
        super(FoundationModel, self).__init__()

        # 1. Load the Meta DINOv2 ViT-Base backbone (vit_base_patch14_dinov2.lvd142m)
        self.backbone = timm.create_model(
            'vit_base_patch14_dinov2.lvd142m',
            pretrained=True,
            num_classes=0,
            img_size=224,
            dynamic_img_size=True
        )

        self.embed_dim = self.backbone.num_features  # Usually 768 for ViT-Base

        # 2. Apply LoRA (Low-Rank Adaptation)
        if use_lora:
            # We target Q, V and Output Projection to handle the geographic/structural shift
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["qkv", "proj"],
                lora_dropout=0.1,
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            print("\n--- LoRA Parameters ---")
            self.backbone.print_trainable_parameters()
        else:
            # Fully freeze backbone parameters when evaluating raw DINOv2
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Standard forward pass for downstream feature extraction.

        Args:
            x (torch.Tensor): Input images [B, 3, 224, 224].

        Returns:
            torch.Tensor: High-dimensional visual embeddings [B, embed_dim=768].
        """
        return self.backbone(x)


class DINOPretrainWrapper(nn.Module):
    """
    Wrapper specifically for the Lightly DINO Self-Supervised Learning (SSL) Training Phase.
    
    Encapsulates both Student and Teacher networks sharing identical architectures:
    - Student: Trained via backpropagation (updating LoRA parameters and projection head).
    - Teacher: Updated strictly via Exponential Moving Average (EMA) of student parameters.

    Args:
        foundation_model (FoundationModel): Base foundation model instance containing LoRA backbone.
    """

    def __init__(self, foundation_model):
        super().__init__()
        self.student_backbone = foundation_model.backbone

        # Re-initialize DINO Projection Head from scratch for the Student
        # Projects 768-d ViT features to high-dimensional categorical distribution (65536)
        self.student_head = DINOProjectionHead(
            input_dim=foundation_model.embed_dim,
            output_dim=65536,  # Standard DINO out_dim
            hidden_dim=2048,
            bottleneck_dim=256,
            batch_norm=False,  # DINO natively avoids BN in the head
            freeze_last_layer=1
        )

        # Recreate the exact same base model architecture for the Teacher
        teacher_base = timm.create_model(
            'vit_base_patch14_dinov2.lvd142m',
            pretrained=False,  # Doesn't matter, we will overwrite weights
            num_classes=0,
            img_size=224,
            dynamic_img_size=True
        )

        # Apply the exact same LoRA config to make the architectures identical
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["qkv", "proj"],
            lora_dropout=0.1,
            bias="none",
        )
        self.teacher_backbone = get_peft_model(teacher_base, lora_config)

        # Recreate a fresh head for teacher
        self.teacher_head = DINOProjectionHead(
            input_dim=foundation_model.embed_dim,
            output_dim=65536,
            hidden_dim=2048,
            bottleneck_dim=256,
            batch_norm=False
        )

        # 2. Copy the exact weights from the student to the teacher via state_dict
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        # Freeze the entire Teacher network (updated strictly via EMA)
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        Student forward pass processing both global and local multi-crop views.
        """
        features = self.student_backbone(x)
        projections = self.student_head(features)
        return projections

    def forward_teacher(self, x):
        """
        Teacher forward pass processing global views only.
        """
        features = self.teacher_backbone(x)
        projections = self.teacher_head(features)
        return projections


class FoundationModelLinearProbe(nn.Module):
    """
    Wraps a frozen Foundation Model with a trainable Linear classification head
    for online streaming linear probing (Experiment 12).

    Args:
        foundation_model (nn.Module): Pretrained/extended Foundation Model backbone.
        num_classes (int): Number of target streaming classes (default: 25).
    """
    def __init__(self, foundation_model: nn.Module, num_classes: int = 25):
        super().__init__()
        self.foundation_model = foundation_model
        # Freeze entire foundation model backbone
        for p in self.foundation_model.parameters():
            p.requires_grad = False
        self.foundation_model.eval()

        embed_dim = getattr(foundation_model, 'embed_dim', 768)
        self.linear_probe = LinearProbe(input_dim=embed_dim, num_classes=num_classes)

    def train(self, mode: bool = True):
        """
        Sets linear probe to training mode while strictly locking the FM backbone in eval mode.
        """
        super().train(mode)
        # Ensure foundation model backbone strictly remains in eval mode
        self.foundation_model.eval()
        return self

    def forward(self, x):
        """
        Forward pass accepting either raw image tensors [B, 3, 224, 224] or precomputed embeddings [B, 768].
        """
        if x.dim() == 4:
            with torch.no_grad():
                features = self.foundation_model(x)
        else:
            features = x
        logits = self.linear_probe(features)
        return {'features': features, 'logits': logits}

