import torch
import torchvision

import src.anchorbox

# from utils import intersection_over_union

SCALE = 208

N_ANCHORS = 9


class YoloLoss(torch.nn.Module):
    def __init__(self, logfn):
        super().__init__()
        self.log = logfn
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.entropy = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(
                [
                    # from get_acnhors script
                    0.0008539709649871904,
                    0.0009398496240601503,
                    0.0006230529595015577,
                    0.0008771929824561404,
                    0.0005668934240362812,
                    0.0012165450121654502,
                    0.00030609121518212427,
                    0.0006277463904582549,
                    0.00031725888324873094,
                    0.0011806375442739079,
                    0.0012135922330097086,
                    0.0004938271604938272,
                    0.0009328358208955224,
                    0.0009505703422053232,
                    7.543753771876886e-05,
                    0.0006724949562878278,
                    0.0009345794392523365,
                    0.0012285012285012285,
                    0.001081081081081081,
                    0.0009025270758122744,
                ]
            )
        )
        self.mse = torch.nn.MSELoss()

        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors, stage):
        anchors = anchors.to(predictions.device)
        batch_size, total_features, h, w = predictions.shape

        # n_anchors = N_ANCHORS
        # features = 25
        # predictions_reshaped = predictions.swapaxes(1, -1).reshape(-1, features)
        # predictions_coords = torch.zeros(batch_size, w, h, n_anchors, features)
        # predictions_coords = src.anchorbox.transform_nn_output_to_coords(
        #     SCALE,
        #     predictions[..., :],
        #     anchor_p_w=anchors[..., 2],
        #     anchor_p_h=anchors[..., 3],
        # )
        target = target.swapaxes(1, -1).reshape(-1, 6)
        predictions = predictions.swapaxes(1, -1).reshape(-1, 25)
        anchors = anchors.swapaxes(1, -1).reshape(-1, 4).repeat(batch_size, 1)
        obj = target[..., 4] == 1
        noobj = target[..., 4] == 0

        # noobject loss
        no_object_loss = self.bce(predictions[..., 4:5][noobj], target[..., 4:5][noobj])
        self.log({f"{stage}_no_object_l": no_object_loss.item()})
        # object loss

        # anchors = anchors.reshape(1, 3, 1, 1, 2)
        # box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        # ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        # object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # object loss
        object_loss = self.bce(predictions[..., 4:5][obj], target[..., 4:5][obj])
        # get iou
        self.log({f"{stage}_object_l": object_loss.item()})
        # class loss
        class_loss = self.entropy(
            (predictions[..., 5:][obj]),
            (target[..., 5][obj].long()),
        )
        self.log({f"{stage}_class_l": class_loss.item()})

        # coordinates
        # predictions[..., 0:2] = torch.sigmoid(predictions[..., 0:2])  # x,y coordinates
        # target[..., 3:5] = torch.log(
        #     (1e-16 + target[..., 3:5] / anchors)
        # )  # width, height coordinates
        box_loss = self.mse(
            predictions[..., 2:4][obj],
            target[..., 2:4][obj],
        )
        self.log({f"{stage}_coords_l": box_loss.item()})

        return (
            self.lambda_noobj * no_object_loss
            + self.lambda_obj * object_loss
            + self.lambda_class * class_loss
            + self.lambda_box * box_loss
        )
