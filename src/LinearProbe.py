from torch import nn


class LinearProbe(nn.Module):
    """
    Single linear classification layer used for probing representation quality.

    Args:
        input_dim (int): Feature dimensionality from backbone (e.g. 768 for DINOv2 ViT-Base).
        num_classes (int): Number of target classes to predict (default: 62).
    """
    def __init__(self, input_dim, num_classes=62):
        super(LinearProbe, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x):
        """Forward pass through linear layer returning classification logits."""
        return self.classifier(x)