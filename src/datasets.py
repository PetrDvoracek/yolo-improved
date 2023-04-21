import torch
import torchvision
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from PIL import Image, ImageFile

# import config
import numpy as np
import os
import pandas as pd
import torch

import src.anchorbox


ImageFile.LOAD_TRUNCATED_IMAGES = True


# def offset_xy(offset_in_cell):
#     return -torch.log((1 - offset_in_cell) / offset_in_cell)


# def offset_wh(w, anchor_w):
#     return torch.log(w / anchor_w)


# def inverse_offset_xy(offset_in_cell):
#     return


class PascalVOC(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        scale=208,
        n_anchors=1,
        resolution=416,
        # transform=None
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        # self.transform = transform
        self.scale = scale
        self.n_anchors = n_anchors
        self.res = resolution
        anch_sizes = [0.75, 0.5, 0.25]
        anch_ratios = [1, 2, 0.5]
        anch_n = len(anch_sizes) + len(anch_ratios) - 1
        self.anchors = src.anchorbox.multibox_prior(
            sizes=anch_sizes, ratios=anch_ratios, imw=scale, imh=scale
        ).reshape(scale, scale, anch_n, 4)
        self.bbox_scale = torch.tensor((resolution, resolution, resolution, resolution))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])

        # [class, x, y, w, h]
        gt_bboxes = np.loadtxt(
            fname=label_path, delimiter=" ", ndmin=2, dtype=np.float32
        )
        gt_bboxes = torch.from_numpy(gt_bboxes)

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        image = image.resize((self.res, self.res)).convert("RGB")
        image = np.array(image)  # H x W x C
        image = (
            torch.from_numpy(image).swapaxes(0, -1).to(torch.float32) / 255
        )  # C x W x H

        features_in_anchor = 4 + 1 + 1  # position + objectness + class
        label = torch.zeros(
            *self.anchors.shape[:-1], features_in_anchor, dtype=torch.float32
        )  # position + objectness + class
        label = label.reshape(self.scale, self.scale, 5, -1)
        for cent_bbox in gt_bboxes:
            label = self.assign_box(label, cent_bbox[1:], box_cls=cent_bbox[0])
        # if self.transform:
        #     augmentations = self.transform(image=image, bboxes=bboxes)  # albumentations
        #     image = augmentations["image"]
        #     bboxes = augmentations["bboxes"]

        label = label.reshape(self.scale, self.scale, -1)
        # label = label.swapaxes(0, 1)  # somewhere is swaped x y axis... TODO
        label = label.swapaxes(0, -1)
        return image, label

    def assign_box(self, label, cent_bbox, box_cls):
        cor_bbox = src.anchorbox.box_center_to_corner(cent_bbox.unsqueeze(0))[0]
        b_x, b_y, w, h = cent_bbox

        c_x = int(b_x * self.scale)
        c_y = int(b_y * self.scale)
        cell_anchors = self.anchors[c_y, c_x]
        ious = torchvision.ops.box_iou(cell_anchors, cor_bbox.unsqueeze(0))[0]
        best_iou_anchor_idx = ious.argsort(descending=True)[0]
        best_anchor = self.anchors[c_x, c_y, best_iou_anchor_idx]

        offset_x_in_cell = (self.scale * b_x) % 1
        offset_y_in_cell = (self.scale * b_y) % 1
        t_x = src.anchorbox.box_coords_center_to_pred(offset_x_in_cell)
        t_y = src.anchorbox.box_coords_center_to_pred(offset_y_in_cell)

        anchor_w, anchor_h = best_anchor[2:4]
        t_w = src.anchorbox.box_coords_size_to_pred(w, anchor_w)
        t_h = src.anchorbox.box_coords_size_to_pred(h, anchor_h)
        label[c_x, c_y, best_iou_anchor_idx, :4] = torch.tensor(
            [
                t_x,
                t_y,
                t_w,
                t_h,
            ]
        )
        label[c_x, c_y, best_iou_anchor_idx, 4] = 1.0
        label[c_x, c_y, best_iou_anchor_idx, 5] = box_cls.item()
        # debug
        # label[cell_idx_x, cell_idx_y, best_iou_anchor_idx, 6:] = cor_bbox
        return label


if __name__ == "__main__":
    IMAGE_SIZE = 416
    SCALE = 26
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
        scale=SCALE,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=None,
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )
    for im, label in data_loader:
        label = label.swapaxes(0, -1).reshape(SCALE, SCALE, 5, -1)
        scale = label.shape[1]
        # positive_boxes = src.anchorbox.assign_anchors_inverse(SCALE, label_abs)
        label = src.anchorbox.transform_nn_output_to_coords(
            scale,
            label,
            dataset.anchors[..., 2],
            dataset.anchors[..., 3],
        )
        positive_boxes = label[label[..., 4] == 1.0]
        positive_boxes = src.anchorbox.box_center_to_corner(positive_boxes)

        im = im.swapaxes(0, -1).numpy()
        im = src.anchorbox.put_bboxes(
            im, (positive_boxes * dataset.res).to(torch.int64).tolist()
        )
        plt.imshow(im)
        plt.savefig("./tmp.png")
        # plt.show()
