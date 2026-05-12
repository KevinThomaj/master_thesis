import timm
from torch import nn
from torchvision import models

class StudentVit(nn.Module):
    def __init__(self,numberOfClasses=62,pretrained=True):
        super().__init__()
        self.numberOfClasses = numberOfClasses
        #Small Vision transformer with 22 Milions of parameters
        self.student = timm.create_model('vit_small_patch16_224', pretrained)

    def forward(self, inputTensor):
        return self.student(inputTensor)


