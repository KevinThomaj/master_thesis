from torch import nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
class Student(nn.Module):
    def __init__(self,numberOfClasses=62):
        super().__init__()
        self.student = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.student.fc = nn.Linear(in_features=512, out_features=numberOfClasses, bias=True)

        self.return_node = {
            'flatten': 'features',
            'fc': 'logits'
        }
        self.student = create_feature_extractor(self.student, self.return_node)


    def forward(self,inputTensor):
        # Returns a dict: {'features': tensor, 'logits': tensor}
        return self.student(inputTensor)