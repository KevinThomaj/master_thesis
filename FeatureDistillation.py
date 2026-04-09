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


class Projector(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self,x):
        raise NotImplementedError("Bo")


#remember to do batch normalization
class LinearBNLogSumDistiller(Projector):
    def __init__(self,dimFeatureStudent,dimFeatureTeacher):
        super().__init__()
        self.model1 = nn.Sequential(
            nn.Linear(dimFeatureStudent,dimFeatureTeacher),
            nn.BatchNorm1d(dimFeatureTeacher)
        )
        self.model2 = nn.BatchNorm1d(dimFeatureStudent)
    def forward(self,student_features,teacher_features):
        projected_student_features = self.model1(student_features)
        projected_teacher_features = self.model2(teacher_features)
        #non per forza deve essere due la potenza, anzi
        bo = (projected_student_features - projected_teacher_features) ** 2
        sum = torch.sum(bo,dim = 1)
        loss = torch.log(sum).mean()
        return loss


#in the code I will need just to do this  distiller = LinearReluDistiller(dim1,dim2)
# loss = distiller(stud_feat, teacher_feat)
# loss.backward()

class LinearReluDistiller(Projector):
    def __init__(self,dimFeatureStudent,dimFeatureTeacher):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dimFeatureStudent,dimFeatureTeacher),
            nn.ReLU()
        )
    def forward(self,student_features,teacher_features):
        projected_student_features = self.model(student_features)
        loss = nn.functional.mse_loss(projected_student_features,teacher_features)
        return loss


class MLPDistiller(Projector):
    def __init__(self,dimFeatureStudent,dimFeatureTeacher,hiddenLayerSize):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dimFeatureStudent,hiddenLayerSize),
            nn.ReLU(),
            nn.Linear(hiddenLayerSize,dimFeatureTeacher),
            nn.ReLU() #forse va messo, forse no
        )
    def forward(self,student_features,teacher_features):
        projected_student_features = self.model(student_features)
        loss = nn.functional.mse_loss(projected_student_features,teacher_features)
        return loss

