import copy

import torch
from torch import nn
class EmaTeacher(nn.Module):
    def __init__(self,student,alpha):
        super().__init__()
        self.alpha = alpha
        # Safely initialize EMA by deepcopying the student
        self.emaTeacher = copy.deepcopy(student)
        # EMA should not track gradients
        for param in self.emaTeacher.parameters():
            param.requires_grad = False



    def update(self,student):
        # Update weights: θ_EMA <- α * θ_EMA + (1-α) * θ_student
        with torch.no_grad():
            for ema_param, student_param in zip(self.emaTeacher.parameters(), student.parameters()):
                ema_param.data.mul_(self.alpha).add_(student_param.data, alpha=1 - self.alpha)

    def predict(self,inputTensor):
        self.emaTeacher.eval()
        with torch.no_grad():
            # Prediction is done by the EMA Teacher
            output = self.emaTeacher(inputTensor)
            return output['logits']