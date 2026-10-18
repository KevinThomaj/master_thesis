import timm
from torch import nn


class StudentVit(nn.Module):
    """
    Lightweight Vision Transformer (ViT-Tiny) Student model.

    Utilizes 'vit_tiny_patch16_224' (~5.7M parameters) as an alternative lightweight backbone
    for on-orbit deployment. It extracts pooled token representations alongside classification logits.

    Args:
        numberOfClasses (int): Number of target classes for classification (default: 62).
        pretrained (bool): Whether to initialize backbone with ImageNet pretrained weights.
    """
    def __init__(self, numberOfClasses=62, pretrained=True):
        super().__init__()
        self.numberOfClasses = numberOfClasses
        # Small Vision transformer with ~5.7M parameters
        # Set num_classes=0 to get raw pooled features directly (num_features = 192)
        self.student = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained, num_classes=0)
        
        # Dynamically create the classifier head using the backbone's feature dimension
        self.fc = nn.Linear(self.student.num_features, numberOfClasses)

    def forward(self, inputTensor):
        """
        Forward pass through the ViT-Tiny student network.

        Args:
            inputTensor (torch.Tensor): Input batch of images [B, 3, 224, 224].

        Returns:
            dict: Dictionary with keys:
                - 'features': Pooled representation [B, 192].
                - 'logits': Raw classification logits [B, numberOfClasses].
        """
        # Extract features
        features = self.student(inputTensor)
        # Classify
        logits = self.fc(features)
        # Return exact dictionary format expected by TrainingManager
        return {'features': features, 'logits': logits}

