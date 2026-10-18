#First approach using a Linear + Batch Normalization + Log Sum Loss (paper Understanding the Effects of Projectors in Knowledge Distillation 2024)
#Second approach using Linear + Relu as activation function + MSE loss (to have non linear effect) paper(Improved Feature Distillation via Projector Ensemble)
#Third approach can be using an MLP with one hidden layer to see how it works(Linear -> Relu -> Linear) and MSE Loss
#Fourth approach Relational KD (no projector)
#Fifth Agnostic architecture distillation



#Using MLP with multiple hidden layer is shown to not have better effect. It is shown that simply
#increasing the depth of the projector does not improve
#the performance of the student and tends to degrade the
#effectiveness of the projector. We hypothesize that with the
#increase of depth, the teacher’s features can be overfitted
#by the projector.(paper Improved Feature Distillation via Projector Ensemble)
import torch
from torch import nn
import torch.nn.functional as F

class Projector(nn.Module):
    """
    Abstract base class for feature projection and distillation modules.
    """
    def __init__(self):
        super().__init__()

    def forward(self, student_features, teacher_features):
        raise NotImplementedError("Subclasses must implement the forward distillation loss computation.")


#remember to do batch normalization
'''
class LinearBNLogSumDistiller(Projector):
    def __init__(self,dimFeatureStudent,dimFeatureTeacher):
        super().__init__()
        self.alpha = 4
        self.model1 = nn.Sequential(
            nn.Linear(dimFeatureStudent,dimFeatureTeacher),
            nn.BatchNorm1d(dimFeatureTeacher,affine=False,eps=1e-4)
        )
        self.model2 = nn.BatchNorm1d(dimFeatureTeacher,affine=False,eps=1e-4)
    def forward(self,student_features,teacher_features):
        projected_student_features = self.model1(student_features)
        projected_teacher_features = self.model2(teacher_features)

        squared_diff = (projected_student_features - projected_teacher_features) ** 2
        sum_squared_diff = torch.sum(squared_diff,dim= 1)
        norm_pow_alpha = sum_squared_diff ** (self.alpha / 2.0)
        loss = torch.log(norm_pow_alpha).mean()

        return loss
'''

#in the code I will need just to do this  distiller = LinearDistiller(dim1,dim2)
# loss = distiller(stud_feat, teacher_feat)
# loss.backward()

class LinearDistiller(Projector):
    """
    Linear feature projector that maps student feature dimension to teacher feature dimension
    and computes Mean Squared Error (MSE) loss.

    Args:
        dimFeatureStudent (int): Feature dimensionality of student model (e.g. 512 for ResNet18, 192 for ViT-Tiny).
        dimFeatureTeacher (int): Feature dimensionality of teacher/foundation model (e.g. 768 for DINOv2 ViT-Base).
    """
    def __init__(self, dimFeatureStudent, dimFeatureTeacher):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dimFeatureStudent, dimFeatureTeacher)
        )

    def forward(self, student_features, teacher_features):
        """
        Projects student features into the teacher representation space and computes MSE loss.

        Args:
            student_features (torch.Tensor): Student features [B, D_student].
            teacher_features (torch.Tensor): Teacher features [B, D_teacher].

        Returns:
            torch.Tensor: Scalar MSE loss value.
        """
        projected_student_features = self.model(student_features)
        loss = nn.functional.mse_loss(projected_student_features, teacher_features)
        return loss


class RKDDistiller(Projector):
    """
    Relational Knowledge Distillation (RKD) distance-wise module.
    
    Instead of projecting representations directly, RKD transfers structural relations
    (pairwise distances between examples in the mini-batch) normalized by the batch mean distance.
    This makes distillation parameter-free and architecture-agnostic.
    """
    def __init__(self):
        super().__init__()
        # No projector needed!

    def forward(self, student_features, teacher_features):
        """
        Computes RKD distance loss using Huber / Smooth L1 loss on normalized pairwise distance matrices.

        Args:
            student_features (torch.Tensor): Student features [B, D_student].
            teacher_features (torch.Tensor): Teacher features [B, D_teacher].

        Returns:
            torch.Tensor: Scalar RKD loss value.
        """
        # Compute pairwise Euclidean distances (no normalisation)
        s_dist = torch.cdist(student_features, student_features, p=2)
        t_dist = torch.cdist(teacher_features, teacher_features, p=2)

        # Compute μ as the average distance in the teacher's batch
        mu = t_dist.mean()

        # Normalise distances by μ
        s_dist_norm = s_dist / mu
        t_dist_norm = t_dist / mu

        # Huber loss on normalised distances
        loss = F.smooth_l1_loss(s_dist_norm, t_dist_norm)
        return loss


class MLPDistiller(Projector):
    """
    Non-linear Multi-Layer Perceptron (MLP) feature projector with a single hidden layer and ReLU activation.
    Computes MSE loss between projected student features and teacher features.

    Args:
        dimFeatureStudent (int): Student feature dimension.
        dimFeatureTeacher (int): Teacher feature dimension (768).
        hiddenLayerSize (int): Dimension of the intermediate hidden layer (e.g. 1024).
    """
    def __init__(self, dimFeatureStudent, dimFeatureTeacher, hiddenLayerSize):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dimFeatureStudent, hiddenLayerSize),
            nn.ReLU(),
            nn.Linear(hiddenLayerSize, dimFeatureTeacher)
        )

    def forward(self, student_features, teacher_features):
        """
        Forward pass through the MLP projector and MSE loss calculation.

        Args:
            student_features (torch.Tensor): Student features [B, D_student].
            teacher_features (torch.Tensor): Teacher features [B, D_teacher].

        Returns:
            torch.Tensor: Scalar MSE loss value.
        """
        projected_student_features = self.model(student_features)
        loss = nn.functional.mse_loss(projected_student_features, teacher_features)
        return loss


