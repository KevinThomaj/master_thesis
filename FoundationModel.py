import torch
import torch.nn as nn
import timm

class FoundationModel(nn.Module):
    def __init__(self):
        super(FoundationModel, self).__init__()

        # Load the Meta DINOv2 ViT-Large backbone pretrained on the LVD-142M dataset
        # Setting num_classes=0 strips the classification head, automatically
        # configuring the model to output flattened global image embeddings.
        self.backbone = timm.create_model(
            'vit_base_patch14_dinov2.lvd142m',
            pretrained=True,
            num_classes=0,
            img_size=224,
        )

        # Freeze weights and set layers like Dropout to evaluation mode
        self.backbone.eval()

    def forward(self, inputTensor):
        with torch.no_grad():
            # Output is a neatly shaped tensor: (Batch_Size, Embedding_Dimension)
            features = self.backbone(inputTensor)
        return features