import numpy as np
import cv2
import torch
import torchvision
import pytorch_lightning as pl
import tqdm
import wandb

import warnings

import src.datasets
import src.utils
import src.config
import src.yolo
import src.loss

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def log_image(im, gt, pred, name):
    wandb.log(
        {
            name: [
                wandb.Image(
                    im,
                    masks={
                        "predictions": {
                            "mask_data": cv2.dilate(
                                cv2.resize(
                                    pred,
                                    (416, 416),
                                    interpolation=cv2.INTER_NEAREST,
                                ).astype(np.uint8),
                                kernel=np.ones((5, 5), np.uint8),
                                iterations=4,
                            ),
                            "class_labels": {0: "bg", 1: "obj"},
                        },
                        "ground_truth": {
                            "mask_data": cv2.dilate(
                                cv2.resize(
                                    gt,
                                    (416, 416),
                                    interpolation=cv2.INTER_NEAREST,
                                ).astype(np.uint8),
                                kernel=np.ones((5, 5), np.uint8),
                                iterations=4,
                            ),
                            "class_labels": {0: "bg", 1: "obj"},
                        },
                    },
                )
            ],
        }
    )


class Trainee(pl.LightningModule):
    def __init__(self, model, lr, weight_decay):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

        self.criterion = src.loss.YoloLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.model.parameters())
        # optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        # )
        return optimizer

    def training_step(self, data, idx):
        inputs, labels = data
        out = self.model(inputs)
        loss = self.criterion(out, labels)
        self.log("train_loss", loss.item())

        if idx % 100 == 1:
            im = inputs[0].detach().cpu().swapaxes(0, -1).numpy()

            gt = labels[0].swapaxes(0, -1).reshape(208, 208, -1).sum(axis=-1)
            gt = gt.to(torch.uint8).detach().cpu().numpy()

            pred = out[0].swapaxes(0, -1).reshape(208, 208, 5, 6)[..., 4]
            pred[pred > 0.9] = 1.0
            pred[pred <= 0.9] = 0.0
            pred = pred.sum(axis=-1)
            pred[pred >= 1.0] = 1.0
            pred[pred < 1.0] = 0.0

            pred = pred.detach().cpu().numpy()

            log_image(im, gt, pred, name="train_images")

        return loss

    def validation_step(self, data, idx):
        inputs, labels = data
        out = self.model(inputs)
        loss = self.criterion(out, labels)
        self.log("val_loss", loss.item())

        if idx % 100 == 1:
            im = inputs[0].detach().cpu().swapaxes(0, -1).numpy()
            gt = labels[0, 0].detach().cpu().swapaxes(0, -1).numpy()
            pred = out[:, 0].argmax(axis=0).detach().cpu().numpy()
            log_image(im, gt, pred, name="val_images")

        return loss


def main():
    DEVICE = "cuda"
    model = src.yolo.YOLOv3(num_classes=1).to(DEVICE)

    dataset = src.datasets.PascalVOC(
        csv_file="/home/pedro/datasets/PASCAL_VOC/100examples.csv",
        img_dir="/home/pedro/datasets/PASCAL_VOC/images",
        label_dir="/home/pedro/datasets/PASCAL_VOC/labels",
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=2,
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )

    trainee = Trainee(
        model, lr=src.config.LEARNING_RATE, weight_decay=src.config.WEIGHT_DECAY
    )

    wandb_exp = wandb.init(
        name="yolo",
        project="yolo",
        entity="petrdvoracek",
        group="test",
        reinit=True,  # just in case of multirun
    )
    wandb_logger = pl.loggers.WandbLogger(experiment=wandb_exp)

    trainer = pl.Trainer(
        gpus=1, precision=16, logger=wandb_logger, accumulate_grad_batches=16
    )
    trainer.fit(trainee, data_loader)

    # for epoch in range(src.config.NUM_EPOCHS):
    #     train_fn(train_loader, model, optimizer, loss_fn, scaler)

    #     if src.config.SAVE_MODEL:
    #         src.utils.save_checkpoint(model, optimizer)

    #     if epoch % 2 == 0 and epoch > 0:
    #         print("on test loader:")
    #         src.utils.check_class_accuracy(model, test_loader, threshold=src.config.CONF_THRESHOLD)

    #         pred_boxes, true_boxes = src.utils.get_evaluation_bboxes(
    #             test_loader,
    #             model,
    #             iou_threshold=src.config.NMS_IOU_THRESH,
    #             anchors=src.config.ANCHORS,
    #             threshold=src.config.CONF_THRESHOLD,
    #         )

    #         mapval = src.utils.mean_average_precision(
    #             pred_boxes,
    #             true_boxes,
    #             iou_threshold=src.config.MAP_IOU_THRESH,
    #             box_format="midpoint",
    #             num_classes=src.config.NUM_CLASSES
    #         )

    #         print(f'mAP: {mapval.item()}')


if __name__ == "__main__":
    main()
