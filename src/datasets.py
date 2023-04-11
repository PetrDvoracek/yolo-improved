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
import src.anchorbox


ImageFile.LOAD_TRUNCATED_IMAGES = True


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
        # [class, xmin, ymin, xmax, ymax]
        gt_bboxes[..., 1:] = src.anchorbox.box_center_to_corner(gt_bboxes[..., 1:])
        # offsets, mask, classes = src.anchorbox.multibox_target(
        #     self.anchors.reshape(1, -1, 4), gt_bboxes.unsqueeze(dim=0)
        # )

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).resize((self.res, self.res)).convert("RGB")
        image = np.array(image)
        image = torch.from_numpy(image).swapaxes(0, -1).to(torch.float32) / 255

        # offsets, mask, classes = src.anchorbox.multibox_target(
        #     self.anchors.reshape(1, -1, 4), gt_bboxes.unsqueeze(dim=0)
        # )
        # offsets = offsets.reshape(self.scale, self.scale, 5, 4)

        label = torch.zeros(
            *self.anchors.shape[:-1], 4 + 1 + 1, dtype=torch.float32
        )  # position + objectness + class
        label = label.reshape(self.scale, self.scale, 5, -1)
        for bbox in gt_bboxes:
            box_cls = bbox[0]
            cent_bbox = src.anchorbox.box_corner_to_center(bbox[1:].unsqueeze(0))[0]
            c_x, c_y, w, h = cent_bbox
            _, x1, y1, x2, y2 = bbox
            grid_idx_x = int(c_x * self.scale)
            grid_idx_y = int(c_y * self.scale)
            cell_anchors = self.anchors[grid_idx_y, grid_idx_x]
            ious = torchvision.ops.box_iou(cell_anchors, bbox[1:].unsqueeze(0))[0]
            best_iou_anchor_idx = ious.argsort(descending=True)[0]
            best_anchor = cell_anchors[best_iou_anchor_idx]
            # offset = torch.tensor(
            #     [
            #         c_x - best_anchor[0],
            #         c_y - best_anchor[1],
            #         w - best_anchor[2],
            #         h - best_anchor[3],
            #     ]
            # )

            # # offset = offsets[grid_idx_x, grid_idx_y, best_iou_anchor_idx]
            # # offset = src.anchorbox.offset_boxes(
            # #     best_anchor.unsqueeze(0), bbox.unsqueeze(0)
            # # )[0]
            # label[grid_idx_x, grid_idx_y, best_iou_anchor_idx, :4] = offset
            label[grid_idx_x, grid_idx_y, best_iou_anchor_idx, 4] = 1.0
            label[grid_idx_x, grid_idx_y, best_iou_anchor_idx, 5] = box_cls.item()
            offset_x_in_cell = (self.scale * c_x) % 1
            offset_y_in_cell = (self.scale * c_y) % 1
            offset_w = torch.log(w) / 0  # prior width TODO
            offset_h = torch.log(h) / 0  # prior height TODO

            # cls_id, x, y, w, h =

        # label[..., :4] = offsets.reshape(1, 208, 208, 5, 4)
        # label[..., 4:5] = classes.reshape(1, 208, 208, 5, 1)

        # labels = []
        # for box in gt_bboxes:
        #     box = box[None, 1:]
        #     iou_with_anchors = src.anchorbox.box_iou(box, self.anchors.reshape(-1, 4))[
        #         0
        #     ]
        #     # best_iou = iou_with_anchors.max()
        #     # anchor_candidate_idxs = (iou_with_anchors == best_iou).nonzero()
        #     # anchors = self.anchors.reshape(-1,4)[anchor_candidate_idxs]
        #     # diffs = torch.abs(box - anchors).sum(dim=-1)
        #     # best_anchor_idx = diffs.argsort()[0]

        #     iou_with_anchors = iou_with_anchors / iou_with_anchors.max()
        #     anchor_mask = iou_with_anchors
        #     anchor_mask[anchor_mask < 1.0] = 0.0
        #     anchor_mask = anchor_mask.to(torch.uint8)
        #     anchor_mask = anchor_mask.reshape(self.anchors.shape[:-1])
        #     label[..., 4][anchor_mask.bool()] = 1.0

        #     iou_indices = iou_with_anchors.argsort(descending=True, dim=0)
        #     best_anchor = self.anchors.reshape(-1, 4)[iou_indices[0].item()]

        # label[..., 0:4] = offsets.reshape(*self.anchors.shape[:-1], 4)
        # label[..., 4] = mask.reshape(*self.anchors.shape[:-2], 1, 1)
        # label[..., 5:] = torch.nn.functional.one_hot(classes, num_classes=20).reshape(
        #     *self.anchors.shape[:-1], 20
        # )

        # if self.transform:
        #     augmentations = self.transform(image=image, bboxes=bboxes)  # albumentations
        #     image = augmentations["image"]
        #     bboxes = augmentations["bboxes"]

        label = label.reshape(self.scale, self.scale, -1)
        label = label.swapaxes(0, 1)  # somewhere is swaped x y axis... TODO
        label = label.swapaxes(0, -1)
        return image, label


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
        bbox = dataset.anchors[
            label.swapaxes(0, -1).reshape(SCALE, SCALE, 5, -1)[..., 4].bool()
        ]
        im = im.swapaxes(0, -1).numpy()
        im = src.anchorbox.put_bboxes(
            im, (bbox * dataset.bbox_scale).to(torch.int64).tolist()
        )
        if label[..., 4].sum() == 0:
            print("now!")
            plt.imshow(im)
            plt.savefig("./tmp.png")
