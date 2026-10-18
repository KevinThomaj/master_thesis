from torch import nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class Student(nn.Module):
    """
    Lightweight ResNet-18 Student model designed for on-orbit Earth Observation deployment.

    Uses torchvision's feature extraction graph to extract both intermediate penultimate
    representations ('features' of dimension 512) and output classification scores ('logits').

    Args:
        numberOfClasses (int): Number of target classes for classification (default: 62).
        pretrained (bool): Whether to initialize backbone with ImageNet pretrained weights.
    """
    def __init__(self, numberOfClasses=62, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.student = models.resnet18(weights=weights)

        # Replace classification head with target number of classes
        self.student.fc = nn.Linear(in_features=512, out_features=numberOfClasses, bias=True)

        # Intercept intermediate pooling representation alongside logits
        self.return_node = {
            'flatten': 'features',
            'fc': 'logits'
        }
        self.student = create_feature_extractor(self.student, self.return_node)

    def forward(self, inputTensor):
        """
        Forward pass through the student network.

        Args:
            inputTensor (torch.Tensor): Input batch of images [B, 3, 224, 224].

        Returns:
            dict: Dictionary with keys:
                - 'features': Penultimate pooled feature embeddings [B, 512].
                - 'logits': Raw classification logits [B, numberOfClasses].
        """
        # Returns a dict: {'features': tensor, 'logits': tensor}
        return self.student(inputTensor)