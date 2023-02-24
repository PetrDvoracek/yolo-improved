import torch
import torch.nn as nn

# from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_f = nn.CrossEntropyLoss()

    def forward(self, predictions, target):
        center_loss = self.loss_f(predictions, target)
        return center_loss