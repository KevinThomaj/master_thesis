import torch
import torch.nn as nn
import timm
import copy
from peft import LoraConfig, get_peft_model
from lightly.models.modules import DINOProjectionHead


class FoundationModel(nn.Module):
    def __init__(self, use_lora=True):
        super(FoundationModel, self).__init__()

        # 1. Load the Meta DINOv2 ViT-Base backbone
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
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Standard forward pass for downstream feature extraction."""
        return self.backbone(x)


class DINOPretrainWrapper(nn.Module):
    """
    Wrapper specifically for the Lightly DINO SSL Training Phase.
    Holds both Student and Teacher networks.
    """

    def __init__(self, foundation_model):
        super().__init__()
        self.student_backbone = foundation_model.backbone

        # Re-initialize DINO Projection Head from scratch for the Student
        self.student_head = DINOProjectionHead(
            input_dim=foundation_model.embed_dim,
            output_dim=65536,  # Standard DINO out_dim
            hidden_dim=2048,
            bottleneck_dim=256,
            batch_norm=False,  # DINO natively avoids BN in the head
            freeze_last_layer=1
        )

        # Recreate the exact same base model architecture
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

        # Recreate a fresh head
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
        """Student forward pass."""
        features = self.student_backbone(x)
        projections = self.student_head(features)
        return projections

    def forward_teacher(self, x):
        """Teacher forward pass."""
        features = self.teacher_backbone(x)
        projections = self.teacher_head(features)
        return projections