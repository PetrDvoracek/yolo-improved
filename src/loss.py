import torch
import torchvision

import src.anchorbox

# from utils import intersection_over_union


class YoloLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.entropy = torch.nn.CrossEntropyLoss()

        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        anchors = anchors.to(predictions.device)
        batch_size = predictions.shape[0]
        target = target.swapaxes(1, -1).reshape(-1, 6)
        predictions = predictions.swapaxes(1, -1).reshape(-1, 25)
        anchors = anchors.swapaxes(1, -1).reshape(-1, 4).repeat(batch_size, 1)
        obj = target[..., 4] == 1
        noobj = target[..., 4] == 0

        # noobject loss
        no_object_loss = self.bce(predictions[..., 4:5][noobj], target[..., 4:5][noobj])
        # object loss
        object_loss = self.bce(predictions[..., 4:5][obj], target[..., 4:5][obj])
        # class loss
        class_loss = self.entropy(
            (predictions[..., 5:][obj]),
            (target[..., 5][obj].long()),
        )
        return (
            self.lambda_noobj * no_object_loss
            + self.lambda_obj * object_loss
            + self.lambda_class * class_loss
        )
