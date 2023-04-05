import torch

# from utils import intersection_over_union


class YoloLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()

        self.noobj = 10
        self.obj = 1

    def forward(self, predictions, target):
        target = target.reshape(-1, 5, 6)
        predictions = predictions.reshape(-1, 5, 6)
        obj = target[..., 4] == 1
        noobj = target[..., 4] == 0
        no_object_loss = self.bce(predictions[..., 3:4][noobj], target[..., 3:4][noobj])
        object_loss = self.bce(predictions[..., 3:4][obj], target[..., 3:4][obj])

        return self.noobj * no_object_loss + self.obj * object_loss
