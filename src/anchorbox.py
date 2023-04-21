import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torchvision

import os

import src.datasets

# import src.config
import src.utils

# source: https://d2l.ai/chapter_computer-vision/anchor.html


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def box_pred_center_to_coords(offset_in_cell, cell_idx, scale):
    """see chapter 2.1 https://pjreddie.com/media/files/papers/YOLOv3.pdf"""
    # return sigmoid(offset_in_cell) + cell_idx
    return (1 / (1 + torch.exp(-torch.tensor(offset_in_cell))) + cell_idx) / scale


def box_coords_center_to_pred(b, cell_idx=None):
    """see chapter 2.1 https://pjreddie.com/media/files/papers/YOLOv3.pdf"""
    # return inverse_sigmoid(b - cell_idx)
    # return -torch.log(((b - cell_idx)/ (1-(cell_idx))))
    if cell_idx is not None:
        # b is offset in cell
        b = b - cell_idx
    eps = 1e-6
    return torch.log(torch.tensor((b) / (1 - b)) + eps)


def box_pred_size_to_coords(t, prior_anchor_size):
    """see chapter 2.1 https://pjreddie.com/media/files/papers/YOLOv3.pdf"""
    return prior_anchor_size * torch.exp(t)


def box_coords_size_to_pred(size, prior_anchor_size):
    """see chapter 2.1 https://pjreddie.com/media/files/papers/YOLOv3.pdf"""
    eps = 1e-6
    return torch.log(size / prior_anchor_size + eps)


def transform_nn_output_to_coords(scale, label, anchor_p_w, anchor_p_h):
    c_x, c_y = torch.meshgrid(torch.arange(0, scale, 1), torch.arange(0, scale, 1))
    c_x = c_x.to(label.device).unsqueeze(-1)
    c_y = c_y.to(label.device).unsqueeze(-1)

    label[..., 0] = box_pred_center_to_coords(label[..., 0], c_x, scale)
    label[..., 1] = box_pred_center_to_coords(label[..., 1], c_y, scale)
    label[..., 2] = box_pred_size_to_coords(label[..., 2], anchor_p_w)
    label[..., 3] = box_pred_size_to_coords(label[..., 3], anchor_p_h)
    return label


# def transform_coords_to_nn_output(scale, coords_center, anchors):
#     coords_corner = box_center_to_corner(coords_center)

#     c_x, c_y, w, h = coords_center
#     cell_idx_x = int(c_x * scale)
#     cell_idx_y = int(c_y * scale)
#     cell_anchors = anchors[cell_idx_x, cell_idx_y]
#     ious = torchvision.ops.box_iou(cell_anchors, coords_corner.unsqueeze(0))[0]
#     best_iou_anchor_idx = ious.argsort(descending=True)[0]
#     best_anchor = anchors[cell_idx_x, cell_idx_y, best_iou_anchor_idx]

#     features_in_out = 4 + 1 + 1  # 4 spatial + objectness + class
#     n_anchors = anchors.shape[2]
#     nn_output_placehold = torch.zeros(scale, scale, n_anchors, features_in_out)
#     pass  # TODO


# def transform_coords_to_nn_output(scale, out):


# def assign_anchors_inverse(scale, label, anchor_p_w, anchor_p_h, threshold):
#     label = transform_nn_output_to_coords(scale, label, anchor_p_w, anchor_p_h)

#     boxes_from_labels = label[label[..., 4] >= threshold]
#     return boxes_from_labels


def multibox_prior(sizes, ratios, imw, imh, device="cpu"):
    """Generate anchor boxes with different shapes centered on each pixel.
    sizes: respresents area
    ratios represents width / height
    example: sizes=[1], ratios=[1] -> anchors with width of imh...
    """
    num_sizes, num_ratios = len(sizes), len(ratios)
    boxes_per_pixel = num_sizes + num_ratios - 1
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / imh  # Scaled steps in y axis
    steps_w = 1.0 / imw  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (torch.arange(imh, device=device) + offset_h) * steps_h
    center_w = (torch.arange(imw, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing="ij")
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = (
        torch.cat(
            (
                size_tensor * torch.sqrt(ratio_tensor[0]),
                sizes[0] * torch.sqrt(ratio_tensor[1:]),
            )
        )
        * imh
        / imw
    )  # Handle rectangular inputs
    h = torch.cat(
        (
            size_tensor / torch.sqrt(ratio_tensor[0]),
            sizes[0] / torch.sqrt(ratio_tensor[1:]),
        )
    )
    # Divide by 2 to get half height and half width
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(imh * imw, 1) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = torch.stack(
        [shift_x, shift_y, shift_x, shift_y], dim=1
    ).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


def put_bboxes(im, bboxes, color=(0, 255, 0)):
    """Show bounding boxes."""

    for box in bboxes:
        im = cv2.rectangle(im, box[0:2], box[2:4], color=color)
    return im


def unravel_indices(
    indices,
    shape,
):
    r"""
    source: https://github.com/pytorch/pytorch/issues/35674
    Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes


def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes


def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets."""
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset


def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)


def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets."""
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox


def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1:
            break
        iou = box_iou(
            boxes[i, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)
        ).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)


def multibox_detection(
    cls_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.009999999
):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = conf < pos_threshold
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat(
            (class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1
        )
        out.append(pred_info)
    return torch.stack(out)


if __name__ == "__main__":
    data_root = "/home/pedro/datasets/PASCAL_VOC"
    train_dataset = src.datasets.PascalVOC(
        "/home/pedro/datasets/PASCAL_VOC/train.csv",
        img_dir=os.path.join(data_root, "images"),
        label_dir=os.path.join(data_root, "labels"),
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=None,
    )

    for data in iter(train_loader):
        im = data[0]
        gt = data[1]
        gt[..., 1:] = box_center_to_corner(gt[..., 1:])
        imw, imh = im.shape[1:]
        anch_w, anch_h = 13, 13

        anch_sizes = [0.75, 0.5, 0.25]
        anch_ratios = [1, 2, 0.5]
        anch_n = len(anch_sizes) + len(anch_ratios) - 1
        anchors = multibox_prior(
            sizes=anch_sizes, ratios=anch_ratios, imw=anch_w, imh=anch_h
        )

        bbox_scale = torch.tensor((imw, imh, imw, imh))
        offsets, mask, classes = multibox_target(anchors, gt.unsqueeze(dim=0))

        im = im.swapaxes(0, -1).numpy()
        # im2show = put_bboxes(
        #     im,
        #     (anchors.reshape(imw, imh, anch_n, 4)[200, 200, ...] * bbox_scale)
        #     .to(torch.int64)
        #     .tolist(),
        # )
        # anchors = anchors * bbox_scale + offsets.reshape(-1, 4)
        anchors = offset_inverse(
            anchors.squeeze(0), offsets.reshape(-1, 4)
        )  # anchors * bbox_scale + offsets.reshape(-1, 4)
        assigned_anchors = (
            anchors.reshape(1, -1)[mask.bool()].reshape(-1, 4) * bbox_scale
        )

        im2show = put_bboxes(im, (assigned_anchors).to(torch.int64).tolist())

        plt.imshow(im2show)
        plt.show()
        # show_bboxes(
        #     fig.axes,
        #     lbl[..., 1:] * bbox_scale,
        #     ["dog", "cat"],
        #     "k",
        # )
        # show_bboxes(fig.axes, anchors[:, 100, ...] * bbox_scale, ["0", "1", "2", "3", "4"])
        # labels = multibox_target(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))
        # show_bboxes()
