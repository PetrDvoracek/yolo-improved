import numpy as np
import cv2
import torch
import torchvision
import pytorch_lightning as pl
import tqdm
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
import os

import src.datasets
import src.utils
import src.anchorbox

import src.config
import src.yolo
import src.loss

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

N_ANCHORS = 9


def log_image_with_boxes(im, gt, pred, name):
    # gt = torch.clip(gt, min=0.0)
    # pred = torch.clip(pred, min=0.0)
    gt = gt.tolist()
    pred = pred.tolist()
    wandb.log(
        {
            name: [
                wandb.Image(
                    im,
                    boxes={
                        "predictions": {
                            "box_data": [
                                {
                                    "position": {
                                        "minX": x[0],
                                        "maxX": x[2],
                                        "minY": x[1],
                                        "maxY": x[3],
                                    },
                                    "class_id": int(x[5]),
                                    "box_caption": f"{src.config.PASCAL_CLASSES[int(x[5])]}",
                                    "scores": {
                                        "objectness": x[4],
                                    },
                                }
                                for x in pred
                            ],
                            "class_labels": {
                                i: k for i, k in enumerate(src.config.PASCAL_CLASSES)
                            },
                        },
                        "ground_truth": {
                            "box_data": [
                                {
                                    "position": {
                                        "minX": x[0],
                                        "maxX": x[2],
                                        "minY": x[1],
                                        "maxY": x[3],
                                    },
                                    "class_id": int(x[5]),
                                    "box_caption": f"{src.config.PASCAL_CLASSES[int(x[5])]}",
                                    "scores": {
                                        "objectness": x[4],
                                    },
                                }
                                for x in gt
                            ],
                            "class_labels": {
                                i: k for i, k in enumerate(src.config.PASCAL_CLASSES)
                            },
                        },
                    },
                )
            ],
        }
    )


def inverse_nn_output(out, scale, n_anch, anchors):
    pred = out[0].swapaxes(0, -1).reshape(scale, scale, n_anch, -1)
    pred[..., 0:2] = torch.sigmoid(pred[..., 0:2])
    x_grid, y_grid = torch.meshgrid(
        torch.arange(pred.shape[0]), torch.arange(pred.shape[1])
    )
    x_grid = torch.stack([x_grid] * 5, axis=2).to(pred.device)
    y_grid = torch.stack([y_grid] * 5, axis=2).to(pred.device)
    pred[..., 0] = pred[..., 0] + x_grid / x_grid.max()
    pred[..., 1] = pred[..., 1] + y_grid / y_grid.max()
    pred[..., 2:4] = torch.exp(pred[..., 2:4])
    pred[..., 4:] = torch.sigmoid(pred[..., 4:])
    return pred


def postproc(pred, thresh, anchors):
    mask = pred[..., 4]
    mask[mask > thresh] = 1.0
    mask[mask <= thresh] = 0.0
    pred_anchors = anchors[mask.bool()]
    pred_boxes = pred[mask.bool()]
    pred_tensor = torch.zeros(len(pred_anchors), 4 + 1 + 1)
    pred_tensor[..., :2] = pred_boxes[..., :2]
    pred_tensor[..., 2:4] = pred_anchors[..., 2:4] * pred_boxes[..., 2:4]
    pred_tensor[..., 4] = pred[mask.bool()][..., 4]
    pred_tensor[..., 5] = pred[mask.bool()][..., 5:].argmax(dim=-1)
    return pred_tensor


class Trainee(pl.LightningModule):
    def __init__(self, model, lr, weight_decay, scale, anchors):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.scale = scale
        self.anchors = anchors

        self.criterion = src.loss.YoloLoss(logfn=wandb.log)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adadelta(self.model.parameters())
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-4, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=50
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }

    def _step(self, data, idx, stage):
        inputs, labels = data
        out = self.model(inputs)

        orig_shape = out.shape
        # out = out.swapaxes(1, -1).reshape(len(out), self.scale, self.scale, 5, -1)
        # out[..., :2] = out[..., :2] / self.scale
        # out = out.reshape(len(out), self.scale, self.scale, -1).swapaxes(1, -1)

        loss = self.criterion(out, labels, self.anchors, stage=stage)
        self.log(f"{stage}_loss", loss.item())

        if idx % 1000 == 1:
            im = inputs[0].detach().cpu().swapaxes(0, -1).numpy()
            label = (
                labels[0]
                .detach()
                .cpu()
                .swapaxes(0, -1)
                .reshape(self.scale, self.scale, N_ANCHORS, -1)
            )
            label2show = label.detach().clone().to(self.anchors.device)
            label_inversed = src.anchorbox.transform_nn_output_to_coords(
                self.scale, label2show, self.anchors[..., 2], self.anchors[..., 3]
            )
            label_inversed = label_inversed[label_inversed[..., 4] == 1.0]
            label_inversed[..., :4] = src.anchorbox.box_center_to_corner(
                label_inversed[..., :4]
            )

            pred = (
                out[0]
                .detach()
                .swapaxes(0, -1)
                .reshape(self.scale, self.scale, N_ANCHORS, -1)
            )
            # pred = pred.swapaxes(0, 1)
            pred[..., 4] = torch.sigmoid(pred[..., 4])
            # pred[..., :2] = pred[..., :2] / self.scale
            pred2show = pred.detach().clone().to(self.anchors.device)
            pred2show[..., 0:2] = 0.0
            pred_inversed = src.anchorbox.transform_nn_output_to_coords(
                self.scale,
                pred2show,
                self.anchors[..., 2],
                self.anchors[..., 3],
                # threshold=0.9,  # do not log unnecessar
            )
            threshold = 0.5
            pred_inversed = pred_inversed[pred_inversed[..., 4] > threshold]
            # max_boxes = 20
            # pred_inversed = pred_inversed[:20]
            pred_inversed[..., :4] = src.anchorbox.box_center_to_corner(
                pred_inversed[..., :4]
            )
            pred_inversed[..., 5] = pred_inversed[..., 5:].argmax(dim=-1)

            nms_reduced_pred = torchvision.ops.nms(
                pred_inversed[..., :4],
                scores=pred_inversed[..., 4],
                iou_threshold=threshold,
            )
            pred_inversed = torch.index_select(pred_inversed, 0, nms_reduced_pred)

            log_image_with_boxes(
                im,
                label_inversed,
                pred_inversed,
                name=f"{stage}_images_nms"
                # gt_tensor,
                # pred_tensor,
                # name=f"{stage}_images_{thresh}",
            )

        return loss

    def training_step(self, data, idx):
        return self._step(data, idx, stage="train")

    def validation_step(self, data, idx):
        return self._step(data, idx, stage="val")


def main():
    DEVICE = "cuda"
    SCALE = 52
    RESOLUTION = 416
    N_INTERPOLATIONS = 5
    # SCALE = 13

    # debug_transform = A.Compose(
    #     [
    #         A.LongestMaxSize(max_size=RESOLUTION),
    #         A.PadIfNeeded(
    #             min_height=RESOLUTION,
    #             min_width=RESOLUTION,
    #             border_mode=cv2.BORDER_CONSTANT,
    #         ),
    #         # A.RandomCrop(32, padding=4),
    #         # A.BBoxSafeRandomCrop(),
    #         A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #         A.OneOf([A.Rotate(30, interpolation=x) for x in [0, 2, 3]]),
    #         A.Cutout(num_holes=30, max_h_size=24, max_w_size=24),
    #         A.HorizontalFlip(),
    #         A.Normalize(
    #             max_pixel_value=255,
    #         ),
    #         ToTensorV2(),
    #     ],
    #     bbox_params=A.BboxParams(
    #         format="yolo", min_visibility=0.0, label_fields=["labels"]
    #     ),
    # )
    # debug_dataset = src.datasets.PascalVOC(
    #     csv_file="/home/pedro/datasets/PASCAL_VOC/100examples.csv",
    #     img_dir="/home/pedro/datasets/PASCAL_VOC/images",
    #     label_dir="/home/pedro/datasets/PASCAL_VOC/labels",
    #     scale=SCALE,
    #     transform=debug_transform,
    # )
    # debug_dataloader = torch.utils.data.DataLoader(
    #     dataset=debug_dataset,
    #     batch_size=16,
    #     num_workers=8,
    #     shuffle=True,
    #     drop_last=False,
    #     persistent_workers=True,
    #     prefetch_factor=1,
    #     pin_memory=True,
    # )

    # inspired from https://github.com/ultralytics/yolov5/blob/f3ee5960671f7d48c2a71cf666a97318661192af/utils/augmentations.py#L22
    train_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=RESOLUTION, interpolation=cv2.INTER_CUBIC),
            A.PadIfNeeded(
                min_height=RESOLUTION,
                min_width=RESOLUTION,
                border_mode=0,
                value=(0, 0, 0),
            ),
            A.OneOf(
                [
                    A.RandomResizedCrop(
                        height=RESOLUTION,
                        width=RESOLUTION,
                        scale=(0.8, 1.0),
                        ratio=(0.9, 1.1),
                        p=1 / N_INTERPOLATIONS,
                        interpolation=x,
                    )
                    for x in range(0, N_INTERPOLATIONS)
                ],
                p=0.5,
            ),
            # my
            # A.VerticalFlip(),
            A.HorizontalFlip(),
            A.OneOf(
                [
                    A.Perspective(interpolation=x, p=1 / N_INTERPOLATIONS)
                    for x in range(0, N_INTERPOLATIONS)
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.Rotate(limit=20, interpolation=x, p=1 / N_INTERPOLATIONS)
                    for x in range(0, N_INTERPOLATIONS)
                ],
                p=0.2,
            ),
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
            A.Cutout(num_holes=30, max_h_size=24, max_w_size=24),
            # yolov5
            A.OneOf(
                [
                    A.Blur(p=1 / 4),
                    A.MedianBlur(p=1 / 4),
                    A.AdvancedBlur(p=1 / 4),
                    A.GaussianBlur(p=1 / 4),
                ],
                p=0.3,
            ),
            A.ToGray(p=0.1),
            A.CLAHE(p=0.1),
            A.RandomGamma(p=0.3),
            A.ImageCompression(quality_lower=70, p=0.3),  # transforms
            A.Normalize(
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo", min_visibility=0.0, label_fields=["labels"]
        ),
    )

    train_dataset = src.datasets.PascalVOC(
        csv_file="/home/pedro/datasets/PASCAL_VOC/train.csv",
        img_dir="/home/pedro/datasets_fast/PASCAL_VOC/images",
        label_dir="/home/pedro/datasets_fast/PASCAL_VOC/labels",
        scale=SCALE,
        transform=train_transform,
        anch_sizes=[
            0.3538200855255127,
            0.17259711027145386,
            0.8528000116348267,
            0.03512848913669586,
            0.5774290561676025,
        ],
        anch_ratios=[
            0.8993720412254333,
            2.395437002182007,
            0.4246741533279419,
            4.431968688964844,
            1.4753663539886475,
        ],
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=128,
        num_workers=64,
        shuffle=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=1,
        pin_memory=True,
    )

    val_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=RESOLUTION, interpolation=cv2.INTER_CUBIC),
            A.PadIfNeeded(
                min_height=RESOLUTION,
                min_width=RESOLUTION,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.Normalize(
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo", min_visibility=0.0, label_fields=["labels"]
        ),
    )
    val_dataset = src.datasets.PascalVOC(
        csv_file="/home/pedro/datasets/PASCAL_VOC/test.csv",
        img_dir="/home/pedro/datasets_fast/PASCAL_VOC/images",
        label_dir="/home/pedro/datasets_fast/PASCAL_VOC/labels",
        scale=SCALE,
        transform=val_transform,
        # anch_sizes=[0.7382153272628784, 0.061571717262268066, 0.343569815158844],
        # anch_ratios=[0.5068705081939697, 2.6602931022644043, 1.199582576751709],
        anch_sizes=[
            0.3538200855255127,
            0.17259711027145386,
            0.8528000116348267,
            0.03512848913669586,
            0.5774290561676025,
        ],
        anch_ratios=[
            0.8993720412254333,
            2.395437002182007,
            0.4246741533279419,
            4.431968688964844,
            1.4753663539886475,
        ],
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=128,
        num_workers=64,
        shuffle=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=1,
        pin_memory=True,
    )

    wandb_exp = wandb.init(
        name="yolo-no-xy-small",
        project="yolo",
        entity="petrdvoracek",
        group="test",
        reinit=True,  # just in case of multirun
    )
    wandb_logger = pl.loggers.WandbLogger(experiment=wandb_exp)

    # pretrained_model = "./pretrained/epoch=0-step=1035.ckpt"
    # pretrained_model = "./pretrained/checkpoint.ckpt"
    pretrained_model = ""
    try:
        trainee = Trainee.load_from_checkpoint(pretrained_model)
    except:
        print("! could not load pretrained model !")
        model = src.yolo.YOLOv3(num_classes=1).to(DEVICE)
        trainee = Trainee(
            model,
            lr=src.config.LEARNING_RATE,
            weight_decay=src.config.WEIGHT_DECAY,
            scale=SCALE,
            anchors=train_dataset.anchors.to(DEVICE),
        )

    trainer = pl.Trainer(
        gpus=1,
        devices=[0],
        precision=32,
        logger=wandb_logger,
        # accumulate_grad_batches=16,
        log_every_n_steps=1,
        callbacks=[
            pl.callbacks.LearningRateMonitor(),
            pl.callbacks.ModelCheckpoint(dirpath="./pretrained", monitor="val_loss"),
        ],
        max_epochs=10_000,
    )
    trainer.fit(trainee, train_dataloader, val_dataloader)
    # trainer.fit(trainee, debug_dataloader)


if __name__ == "__main__":
    main()
