import numpy as np
import cv2
import torch
import torchvision
import pytorch_lightning as pl
import tqdm
import wandb

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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


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
        optimizer = torch.optim.Adadelta(self.model.parameters())
        # optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        # )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
        return optimizer

    def _step(self, data, idx, stage):
        inputs, labels = data
        out = self.model(inputs)

        orig_shape = out.shape
        # out = out.swapaxes(1, -1).reshape(len(out), self.scale, self.scale, 5, -1)
        # out[..., :2] = out[..., :2] / self.scale
        # out = out.reshape(len(out), self.scale, self.scale, -1).swapaxes(1, -1)

        loss = self.criterion(out, labels, self.anchors)
        self.log(f"{stage}_loss", loss.item())

        if idx % 100 == 1:
            im = inputs[0].detach().cpu().swapaxes(0, -1).numpy()
            label = (
                labels[0]
                .detach()
                .cpu()
                .swapaxes(0, -1)
                .reshape(self.scale, self.scale, 5, -1)
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
                out[0].detach().swapaxes(0, -1).reshape(self.scale, self.scale, 5, -1)
            )
            # pred = pred.swapaxes(0, 1)
            pred[..., 4] = torch.sigmoid(pred[..., 4])
            # pred[..., :2] = pred[..., :2] / self.scale
            pred2show = pred.detach().clone().to(self.anchors.device)
            pred_inversed = src.anchorbox.transform_nn_output_to_coords(
                self.scale,
                pred2show,
                self.anchors[..., 2],
                self.anchors[..., 3],
                # threshold=0.9,  # do not log unnecessar
            )
            threshold = 0.7
            pred_inversed = pred_inversed[pred_inversed[..., 4] > threshold]
            # max_boxes = 20
            # pred_inversed = pred_inversed[:20]
            pred_inversed[..., :4] = src.anchorbox.box_center_to_corner(
                pred_inversed[..., :4]
            )
            pred_inversed[..., 5] = pred_inversed[..., 5:].argmax(dim=-1)

            log_image_with_boxes(
                im,
                label_inversed,
                pred_inversed,
                name=f"{stage}_images"
                # gt_tensor,
                # pred_tensor,
                # name=f"{stage}_images_{thresh}",
            )

            labeled_anchors = label.detach().clone().to(label.device)
            labeled_anchors[..., :4] = self.anchors
            labeled_anchors = labeled_anchors[labeled_anchors[..., 4] == 1.0]
            labeled_anchors[..., :4] = src.anchorbox.box_center_to_corner(
                labeled_anchors[..., :4]
            )

            pred_anchors = pred.detach().clone().to(pred.device)
            pred_anchors[..., :4] = self.anchors
            pred_anchors[..., 5] = pred_anchors[..., 5:].argmax(dim=-1)
            pred_anchors = pred_anchors[pred[..., 4] > threshold]
            pred_anchors[..., :4] = src.anchorbox.box_center_to_corner(
                pred_anchors[..., :4]
            )

            log_image_with_boxes(
                im,
                labeled_anchors,
                pred_anchors,
                name=f"{stage}_anchors",
            )

        return loss

    def training_step(self, data, idx):
        return self._step(data, idx, stage="train")

    def validation_step(self, data, idx):
        return self._step(data, idx, stage="val")


def main():
    DEVICE = "cuda"
    SCALE = 208
    # SCALE = 13

    model = src.yolo.YOLOv3(num_classes=1).to(DEVICE)

    debug_dataset = src.datasets.PascalVOC(
        csv_file="/home/pedro/datasets/PASCAL_VOC/100examples.csv",
        img_dir="/home/pedro/datasets/PASCAL_VOC/images",
        label_dir="/home/pedro/datasets/PASCAL_VOC/labels",
        scale=SCALE,
    )
    debug_dataloader = torch.utils.data.DataLoader(
        dataset=debug_dataset,
        batch_size=16,
        num_workers=8,
        shuffle=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=1,
        pin_memory=True,
    )

    train_dataset = src.datasets.PascalVOC(
        csv_file="/home/pedro/datasets/PASCAL_VOC/train.csv",
        img_dir="/home/pedro/datasets/PASCAL_VOC/images",
        label_dir="/home/pedro/datasets/PASCAL_VOC/labels",
        scale=SCALE,
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=16,
        num_workers=8,
        shuffle=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=1,
        pin_memory=True,
    )

    val_dataset = src.datasets.PascalVOC(
        csv_file="/home/pedro/datasets/PASCAL_VOC/test.csv",
        img_dir="/home/pedro/datasets/PASCAL_VOC/images",
        label_dir="/home/pedro/datasets/PASCAL_VOC/labels",
        scale=SCALE,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=16,
        num_workers=8,
        shuffle=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=1,
        pin_memory=True,
    )

    wandb_exp = wandb.init(
        name="yolo",
        project="yolo",
        entity="petrdvoracek",
        group="test",
        reinit=True,  # just in case of multirun
    )
    wandb_logger = pl.loggers.WandbLogger(experiment=wandb_exp)

    trainee = Trainee(
        model,
        lr=src.config.LEARNING_RATE,
        weight_decay=src.config.WEIGHT_DECAY,
        scale=SCALE,
        anchors=train_dataset.anchors.to(DEVICE),
    )

    trainer = pl.Trainer(
        gpus=1,
        devices=[2],
        precision=32,
        logger=wandb_logger,
        # accumulate_grad_batches=16,
        log_every_n_steps=1,
        callbacks=[pl.callbacks.LearningRateMonitor()],
    )
    # trainer.fit(trainee, train_dataloader, val_dataloader)
    trainer.fit(trainee, debug_dataloader)


if __name__ == "__main__":
    main()
