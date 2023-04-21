import torch
import torchvision

import src.anchorbox

# from utils import intersection_over_union


class YoloLoss(torch.nn.Module):
    def __init__(self, logfn):
        super().__init__()
        self.log = logfn
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.entropy = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()

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
        self.log({"no_object_l": no_object_loss.item()})
        # object loss

        # anchors = anchors.reshape(1, 3, 1, 1, 2)
        # box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        # ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        # object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # object loss
        object_loss = self.bce(predictions[..., 4:5][obj], target[..., 4:5][obj])
        self.log({"object_l": object_loss.item()})
        # class loss
        class_loss = self.entropy(
            (predictions[..., 5:][obj]),
            (target[..., 5][obj].long()),
        )
        self.log({"class_l": class_loss.item()})

        # coordinates
        # predictions[..., 0:2] = torch.sigmoid(predictions[..., 0:2])  # x,y coordinates
        # target[..., 3:5] = torch.log(
        #     (1e-16 + target[..., 3:5] / anchors)
        # )  # width, height coordinates
        box_loss = self.mse(
            predictions[..., 0:4][obj],
            target[..., 0:4][obj],
        )
        self.log({"coords_l": box_loss.item()})

        return (
            self.lambda_noobj * no_object_loss
            + self.lambda_obj * object_loss
            + self.lambda_class * class_loss
            + self.lambda_box * box_loss
        )
