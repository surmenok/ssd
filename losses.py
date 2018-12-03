"""
Based on code from https://github.com/fastai/fastai/blob/master/courses/dl2/pascal-multi.ipynb
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.nn import functional as F
from torch import tensor as T
from custom_types import TensorType


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels

    def forward(self, predicted: TensorType, target: TensorType):
        """
        :param predicted: Type: float32. Shape: (batch_size, anchors, classes)
        :param target: Type: int64. Shape: (batch_size, boxes)
        :return:
        """
        one_hot_target = self.one_hot_encoding(target, self.num_labels + 1)
        one_hot_target = one_hot_target[:, :-1]  # Remove the background class
        one_hot_target = V(one_hot_target.contiguous(), requires_grad=False)
        predicted = predicted[:, :-1]
        weight = self.get_weight(predicted, target)
        return F.binary_cross_entropy_with_logits(predicted, one_hot_target, weight,
                                                  size_average=False) / self.num_labels

    def one_hot_encoding(self, labels, num_labels):
        return torch.eye(num_labels, device=labels.device)[labels.data]

    def get_weight(self, predicted, target):
        # This is a placeholder which will be used for focal loss later
        return None


def filter_ground_truth(boxes, classes):
    heights = boxes[:, 2] - boxes[:, 0]
    to_keep = (heights > 0).nonzero()[:, 0]
    filtered_boxes = boxes[to_keep.detach()]
    filtered_classes = classes[to_keep.detach()]
    return filtered_boxes, filtered_classes


def intersect(boxes1: TensorType, boxes2: TensorType) -> TensorType:
    """
    Finds overlap between all boxes in boxes1 and all boxes in boxes2

    Input shape: (boxes, 4)
    The last dimension: top, left, bottom, right

    Output: areas of intersections.
    Output shape: (boxes1, boxes2)
    """
    max_top_left = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    min_bottom_right = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    dimensions = torch.clamp(min_bottom_right - max_top_left, min=0)
    return dimensions[:, :, 0] * dimensions[:, :, 1]


def area(boxes):
    return (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])


def jaccard_overlap(boxes1, boxes2):
    areas1 = area(boxes1)
    areas2 = area(boxes2)
    intersects = intersect(boxes1, boxes2)
    unions = areas1[:,None] + areas2[None,:] - intersects
    return intersects.float() / unions.float()


def create_anchors(grid_size: int=4):
    anchor_size = 1. / grid_size

    # Centers
    anchor_center_coords = np.linspace(anchor_size / 2, 1 - anchor_size / 2, grid_size)
    anchor_center_x = np.repeat(anchor_center_coords, grid_size).reshape(-1, 1)
    anchor_center_y = np.tile(anchor_center_coords, grid_size).reshape(-1, 1)
    anchor_centers = np.concatenate([anchor_center_x, anchor_center_y], axis=1)

    # Heights and widths
    anchor_hw = np.ones((grid_size ** 2, 2)) * anchor_size

    # anchors = V(T(np.concatenate([anchor_centers, anchor_hw], axis=1)), requires_grad=False).float()
    anchors = T(np.concatenate([anchor_centers, anchor_hw], axis=1)).float()

    return anchors


def box_hw_to_corners(boxes):
    top_left = boxes[:,:2] - boxes[:,2:] / 2
    bottom_right = boxes[:,:2] + boxes[:,2:] / 2
    return torch.cat([top_left, bottom_right], dim=1)


def activation_to_bbox_corners(activation, anchors, anchor_size):
    """
    Convert outputs of bbox head of the model for one example to coordinates
    of corners of bounding boxes in image frame, where coordinates are from 0 to 1.
    :param activation: Type: float32. Shape: (anchors, 4).
        Last dimension: (center top, center left, height, width).
        Center coordinates (-1; 1) range maps to (-0.5*anchor_height; 0.5*anchor_width) relative to the anchor center
        Height and width (-1; 1) range maps to (0.5; 1.5)*anchor_box height and width.
    :param anchors: Type: float32. Shape: (anchors, 4)
    :param anchor_size: Size of an anchor box relative to the image. Type: float.
    :return: Type: float32. Shape: (anchors, 4)
    """
    # tanh to squeeze values into range (-1;1)
    activation_tanh = torch.tanh(activation)
    #
    activation_centers = (activation_tanh[:, :2] / 2 * anchor_size) + anchors[:, :2]
    activation_hw = (activation_tanh[:, 2:] / 2 + 1) * anchors[:, 2:]
    return box_hw_to_corners(torch.cat([activation_centers, activation_hw], dim=1))


def map_ground_truth(bounding_boxes, anchor_boxes, threshold=0.5):
    overlaps = jaccard_overlap(bounding_boxes, anchor_boxes)
    best_bbox_overlaps, best_bbox_ids = overlaps.max(1)
    overlaps, bbox_ids = overlaps.max(0)
    overlaps[best_bbox_ids] = 2
    for bbox_id, anchor_id in enumerate(best_bbox_ids):
        bbox_ids[anchor_id] = bbox_id
    is_positive = overlaps > threshold
    return is_positive, bbox_ids


class SSDLoss:
    def __init__(self, anchors, anchor_size, image_size, num_classes):
        self.anchors = anchors
        self.anchor_corners = box_hw_to_corners(anchors)
        self.anchor_size = anchor_size
        self.image_size = image_size
        self.num_classes = num_classes

        self.calculate_classification_loss = BinaryCrossEntropyLoss(num_classes)

    def calculate_example_loss(self, y_boxes, y_classes, boxes_activation, classes_activation):
        # TODO: This line will change when width and height are different
        y_boxes = y_boxes / self.image_size

        y_boxes_filtered, y_classes_filtered = filter_ground_truth(y_boxes, y_classes)
        is_positive, bbox_ids = map_ground_truth(y_boxes_filtered, self.anchor_corners)

        positive_anchor_ids = torch.nonzero(is_positive)[:, 0]

        ground_truth_bboxes = y_boxes_filtered[bbox_ids]
        ground_truth_classes = y_classes[bbox_ids]

        is_negative = is_positive == 0
        ground_truth_classes[is_negative] = self.num_classes

        boxes_activation_corners = activation_to_bbox_corners(boxes_activation, self.anchors, self.anchor_size)

        diff = ground_truth_bboxes[positive_anchor_ids] - boxes_activation_corners[positive_anchor_ids]
        localization_loss = diff.abs().mean()

        classification_loss = self.calculate_classification_loss(classes_activation, ground_truth_classes)

        return localization_loss, classification_loss

    def batch_losses(self, predicted, target):
        # predicted: a tuple (locations, classes)
        #   locations.shape: (batch_size, num_anchors, 4*k)
        #   classes.shape: (batch_size, num_anchors, k * (num_classes + 1))
        # target: list of tuples (one tuple for each example in a batch)
        #   tuple[0]: ground truth boxes. float32 tensor of shape (num_objects, 4)
        #   tuple[1]: ground truth classes. long tensor of shape (num_objects,)

        # TODO: Verify that batch_size is equal for predicted and target

        classification_loss_sum = 0
        localization_loss_sum = 0
        for i in range(len(target)):
            boxes_activation = predicted[0][i]
            classes_activation = predicted[1][i]
            y_boxes = target[i][0]
            y_classes = target[i][1]
            c, l = self.calculate_example_loss(y_boxes, y_classes, boxes_activation, classes_activation)
            classification_loss_sum += c
            localization_loss_sum += l
        return classification_loss_sum / len(target), localization_loss_sum / len(target)

    def loss(self, predicted, target):
        # predicted: a tuple (locations, classes)
        #   locations.shape: (batch_size, num_anchors, 4*k)
        #   classes.shape: (batch_size, num_anchors, k * (num_classes + 1))
        # target: list of tuples (one tuple for each example in a batch)
        #   tuple[0]: ground truth boxes. float32 tensor of shape (num_objects, 4)
        #   tuple[1]: ground truth classes. long tensor of shape (num_objects,)

        classification_loss, localization_loss = self.batch_losses(predicted, target)
        return 10 * classification_loss + localization_loss
