import torch
import torchvision
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from PIL import Image, ImageDraw, ImageFile

# import config
import numpy as np
import os
import pandas as pd
import torch

import PIL
from src import utils

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PascalVOC(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, scale=208, n_anchors=1,transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.scale = scale
        self.n_anchors = n_anchors

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(  # [class, x, y, w, h] -> [x, y, w, h, class]
            np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1
        ).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)  # albumentations
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        targets = torch.zeros(
            3, self.scale, self.scale # 3 be
        ) # p, w, h of the center

        # TODO make for more anchors
        for box in bboxes:
            x, y, _, _, class_label = box

            i, j = int(self.scale * y), int(self.scale * x)
            box_taken = targets[0, i, j]

            if not box_taken:
                targets[0, i, j] = 1
                x_cell, y_cell = (
                    self.scale * x - j,
                    self.scale * y - i,
                )  # both between [0, 1]
                center_coordinates = torch.tensor([x_cell, y_cell])
                targets[1:3, i, j] = center_coordinates

        return image, targets


if __name__ == "__main__":
    IMAGE_SIZE = 416
    test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_SIZE),
            A.PadIfNeeded(
                min_height=IMAGE_SIZE,
                min_width=IMAGE_SIZE,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.Normalize(
                mean=[0, 0, 0],
                std=[1, 1, 1],
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
    )
    dataset = PascalVOC(
        csv_file="/home/pedro/datasets/PASCAL_VOC/100examples.csv",
        img_dir="/home/pedro/datasets/PASCAL_VOC/images",
        label_dir="/home/pedro/datasets/PASCAL_VOC/labels",
        transform=test_transforms,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=4,
        num_workers=1,
        shuffle=False,
        drop_last=False,
    )
    for im, label in data_loader:
        plt.imshow(utils.draw_centers(im[0], label[0]))
        plt.show()
