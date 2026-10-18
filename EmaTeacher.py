import copy
import torch
from torch import nn


class EmaTeacher(nn.Module):
    """
    Exponential Moving Average (EMA) Teacher.

    Maintains a shadow copy of the student network whose weights are slowly updated
    as a moving average of the active student model:
        θ_EMA <- α * θ_EMA + (1 - α) * θ_student

    This temporal ensembling smooths predictions, stabilizes feature extraction,
    and reduces prediction noise during streaming adaptation.

    Args:
        student (nn.Module): The active student model to track.
        alpha (float): Momentum coefficient (typically 0.99 or 0.999).
    """
    def __init__(self, student, alpha):
        super().__init__()
        self.alpha = alpha
        # Safely initialize EMA by deepcopying the student
        self.emaTeacher = copy.deepcopy(student)
        # EMA should not track gradients
        for param in self.emaTeacher.parameters():
            param.requires_grad = False

    def update(self, student):
        """
        Updates the EMA weights and copies BatchNorm running statistics from the student.

        Args:
            student (nn.Module): The current student model after optimizer step.
        """
        # Update weights: θ_EMA <- α * θ_EMA + (1-α) * θ_student
        with torch.no_grad():
            for ema_param, student_param in zip(self.emaTeacher.parameters(), student.parameters()):
                ema_param.data.mul_(self.alpha).add_(student_param.data, alpha=1 - self.alpha)
            # Update buffers (Crucial for BatchNorm)
            for ema_buffer, student_buffer in zip(self.emaTeacher.buffers(), student.buffers()):
                ema_buffer.data.copy_(student_buffer.data)

    def predict(self, inputTensor):
        """
        Computes forward prediction logits using the frozen EMA weights.

        Args:
            inputTensor (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: Classification logits.
        """
        self.emaTeacher.eval()
        with torch.no_grad():
            # Prediction is done by the EMA Teacher
            output = self.emaTeacher(inputTensor)
            return output['logits']