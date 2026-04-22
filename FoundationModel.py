"""
import torch
from mpmath.calculus.extrapolation import standardize
from torch import nn
from terratorch import BACKBONE_REGISTRY
import torch.nn.functional as F


class FoundationModel(nn.Module):
    def __init__(self,name='terramind_v1_base'):
        super().__init__()
        self.FM = BACKBONE_REGISTRY.build(
                        name = name,
                        pretrained=True,
                        modalities=['RGB'],
                        )
        self.FM.eval()


    def forward(self,inputTensor):

        with torch.no_grad():
            features = self.FM(inputTensor)[-1]
            # Ensure features are flattened to (Batch, Embedding dim)
            features = features.mean(dim = 1)
            #features = F.normalize(features, p=2, dim=1)
            return features
"""
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download


class FoundationModel(nn.Module):
    def __init__(self):

        super(FoundationModel, self).__init__()
        HF_REPO_ID    = "samarkhanna/ExPLoRA"
        FILENAME      = "explora_dinov2_fmow_rgb/explora_dinov2_vit_base_fmow_rgb_encoder_only.pth"
        ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=FILENAME, local_dir="./checkpoints")

        # Load the standard DINOv2 architecture from Meta's hub.
        # Set pretrained=False so you don't waste time/RAM downloading the
        # standard natural-image weights. We only need the empty architecture.
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14',
            img_size=224,
            pretrained=False
        )

        # Load the ExPLoRA satellite weights from your downloaded file.
        # The paper provides "Encoder-only weights" where the LoRA adapters
        # are already merged, making them directly compatible with this backbone!
        state_dict = torch.load(explora_checkpoint_path, map_location="cpu")

        # (Optional safety check: sometimes saved checkpoints wrap the weights in a 'model' key)
        if 'model' in state_dict:
            state_dict = state_dict['model']

        # FIX THE MASK TOKEN ERROR
        # Use strict=False to tell PyTorch it is okay that the checkpoint
        # doesn't contain the pre-training 'mask_token'.
        self.backbone.load_state_dict(state_dict, strict=False)

        self.backbone.eval()

    def forward(self, inputTensor):
        # Extract the [CLS] token without tracking gradients
        with torch.no_grad():
            output = self.backbone(inputTensor)
        return output
