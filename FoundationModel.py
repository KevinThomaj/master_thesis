import torch
from torch import nn
from terratorch import BACKBONE_REGISTRY

class FoundationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.FM = FM = BACKBONE_REGISTRY.build(
                        'terramind_v1_tiny_tim', pretrained=True,
                        modalities=['RGB'],
                        )
        self.FM.eval()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self,inputTensor):
        #ricordare di flattare l'output perchè, vogliamo un vettore non una matrice (per singola immagine)
        with torch.no_grad():
            features = self.FM(inputTensor)[-1]
            # Ensure features are flattened to (Batch, Channels)
            if len(features.shape) > 2:
                features = self.pool(features).flatten(1)
            return features