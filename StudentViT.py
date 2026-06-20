import timm
from torch import nn

class StudentVit(nn.Module):
    def __init__(self, numberOfClasses=62, pretrained=True):
        super().__init__()
        self.numberOfClasses = numberOfClasses
        # Small Vision transformer with 22 Millions of parameters
        # Set num_classes=0 to get raw pooled features directly
        self.student = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=0)
        
        # Dynamically create the classifier head using the backbone's feature dimension
        self.fc = nn.Linear(self.student.num_features, numberOfClasses)

    def forward(self, inputTensor):
        # Extract features
        features = self.student(inputTensor)
        # Classify
        logits = self.fc(features)
        # Return exact dictionary format expected by TrainingManager
        return {'features': features, 'logits': logits}
