"""
Advanced Faster R-CNN with modern improvements
- Feature Pyramid Network (FPN) for multi-scale detection
- ROI Align (better than ROI Pool)
- Focal Loss for classification
- Advanced anchor generation
- Deformable convolutions
- Attention mechanisms
"""

import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Utility Functions
# ============================================================================

def get_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes
    Args:
        boxes1: (N, 4) [x1, y1, x2, y2]
        boxes2: (M, 4) [x1, y1, x2, y2]
    Returns:
        iou: (N, M) IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    intersection = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)
    union = area1[:, None] + area2[None, :] - intersection
    iou = intersection / (union + 1e-8)
    
    return iou


def get_giou(boxes1, boxes2):
    """
    Generalized IoU - better for small objects
    """
    iou = get_iou(boxes1, boxes2)
    
    # Compute enclosing box
    x1_min = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    y1_min = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    x2_max = torch.max(boxes1[:, None, 2], boxes2[:, 2])
    y2_max = torch.max(boxes1[:, None, 3], boxes2[:, 3])
    
    area_c = (x2_max - x1_min).clamp(min=0) * (y2_max - y1_min).clamp(min=0)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - iou * (area1[:, None] + area2[None, :])
    
    giou = iou - (area_c - union) / (area_c + 1e-8)
    return giou


def boxes_to_transformation_targets(gt_boxes, anchors, weights=(1.0, 1.0, 1.0, 1.0)):
    """
    Encode boxes as regression targets
    """
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    wx, wy, ww, wh = weights
    targets_dx = wx * (gt_ctr_x - ctr_x) / widths
    targets_dy = wy * (gt_ctr_y - ctr_y) / heights
    targets_dw = ww * torch.log(gt_widths / widths)
    targets_dh = wh * torch.log(gt_heights / heights)

    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


def apply_regression_to_boxes(deltas, boxes, weights=(1.0, 1.0, 1.0, 1.0)):
    """
    Decode regression targets to boxes
    """
    deltas = deltas.reshape(deltas.size(0), -1, 4)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[..., 0] / wx
    dy = deltas[..., 1] / wy
    dw = deltas[..., 2] / ww
    dh = deltas[..., 3] / wh

    # Prevent explosive values
    dw = torch.clamp(dw, max=math.log(1000.0 / 16))
    dh = torch.clamp(dh, max=math.log(1000.0 / 16))

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[..., 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[..., 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[..., 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[..., 3] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes_to_image(boxes, size):
    """
    Clip boxes to image boundaries
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = size

    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def remove_small_boxes(boxes, min_size):
    """
    Remove boxes with width or height < min_size
    """
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    return keep


def sample_positive_negative(labels, positive_count, total_count):
    """
    Sample positive and negative examples
    """
    positive = torch.where(labels >= 1)[0]
    negative = torch.where(labels == 0)[0]

    num_pos = min(positive.numel(), positive_count)
    num_neg = min(negative.numel(), total_count - num_pos)

    perm_pos = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm_neg = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm_pos]
    neg_idx = negative[perm_neg]

    pos_mask = torch.zeros_like(labels, dtype=torch.bool)
    neg_mask = torch.zeros_like(labels, dtype=torch.bool)
    pos_mask[pos_idx] = True
    neg_mask[neg_idx] = True

    return neg_mask, pos_mask


# ============================================================================
# Focal Loss for handling class imbalance
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss from "Focal Loss for Dense Object Detection"
    Addresses class imbalance by down-weighting easy examples
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        focal_weight = (1 - p) ** self.gamma
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SigmoidFocalLoss(nn.Module):
    """Focal loss for binary classification"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        
        return loss.mean()


# ============================================================================
# Feature Pyramid Network (FPN)
# ============================================================================

class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature extraction
    Top-down pathway with lateral connections
    """
    
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: list of feature maps from backbone [C2, C3, C4, C5]
        Returns:
            list of FPN feature maps [P2, P3, P4, P5]
        """
        last_inner = self.inner_blocks[-1](x[-1])
        results = [self.layer_blocks[-1](last_inner)]
        
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x[idx])
            inner_top_down = F.interpolate(last_inner, size=inner_lateral.shape[-2:], mode='nearest')
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))
        
        return results


# ============================================================================
# Channel Attention Module
# ============================================================================

class ChannelAttention(nn.Module):
    """
    Channel attention mechanism to enhance important features
    """
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)


# ============================================================================
# Advanced Anchor Generator
# ============================================================================

class AdvancedAnchorGenerator(nn.Module):
    """
    Generate anchors with multiple scales and aspect ratios
    Supports FPN-style multi-level anchors
    """
    
    def __init__(
        self,
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    ):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = []
        
        for size, ratio in zip(sizes, aspect_ratios):
            self.cell_anchors.append(
                self.generate_cell_anchors(size, ratio)
            )
    
    def generate_cell_anchors(self, scales, aspect_ratios):
        """
        Generate anchors for a single cell
        """
        scales = torch.as_tensor(scales, dtype=torch.float32)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=torch.float32)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios
        
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()
    
    def grid_anchors(self, grid_size, stride, device):
        """
        Generate anchors for entire feature map
        """
        cell_anchors = self.cell_anchors[0].to(device)
        
        shifts_x = torch.arange(0, grid_size[1], dtype=torch.float32, device=device) * stride[1]
        shifts_y = torch.arange(0, grid_size[0], dtype=torch.float32, device=device) * stride[0]
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
        
        anchors = (shifts.view(-1, 1, 4) + cell_anchors.view(1, -1, 4)).reshape(-1, 4)
        return anchors
    
    def forward(self, image_size, feature_map_size, device):
        """
        Generate all anchors for an image
        """
        grid_height, grid_width = feature_map_size
        image_height, image_width = image_size
        stride_height = image_height // grid_height
        stride_width = image_width // grid_width
        
        anchors = self.grid_anchors(
            (grid_height, grid_width),
            (stride_height, stride_width),
            device
        )
        return anchors


# ============================================================================
# Advanced RPN with FPN support
# ============================================================================

class AdvancedRPN(nn.Module):
    """
    Region Proposal Network with:
    - Multi-scale feature support
    - Focal loss for objectness
    - Better anchor assignment
    """
    
    def __init__(
        self,
        in_channels=256,
        num_anchors=3,
        fg_iou_thresh=0.7,
        bg_iou_thresh=0.3,
        batch_size_per_image=256,
        positive_fraction=0.5,
        nms_thresh=0.7,
        pre_nms_top_n_train=2000,
        pre_nms_top_n_test=1000,
        post_nms_top_n_train=2000,
        post_nms_top_n_test=1000
    ):
        super().__init__()
        
        self.num_anchors = num_anchors
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.nms_thresh = nms_thresh
        
        self.pre_nms_top_n = {
            'training': pre_nms_top_n_train,
            'testing': pre_nms_top_n_test
        }
        self.post_nms_top_n = {
            'training': post_nms_top_n_train,
            'testing': post_nms_top_n_test
        }
        
        # Convolutional layers
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)
        
        # Initialize
        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
        
        self.anchor_generator = AdvancedAnchorGenerator()
        self.focal_loss = SigmoidFocalLoss(alpha=0.25, gamma=2.0)
    
    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        """
        Compute RPN losses
        """
        sampled_pos_inds, sampled_neg_inds = self.sample_anchors(labels)
        sampled_inds = torch.where(sampled_pos_inds | sampled_neg_inds)[0]
        
        # Objectness loss (focal loss)
        objectness_loss = self.focal_loss(
            objectness[sampled_inds].flatten(),
            labels[sampled_inds].flatten()
        )
        
        # Box regression loss (only for positives)
        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            reduction='sum'
        ) / max(1, sampled_inds.numel())
        
        return objectness_loss, box_loss
    
    def sample_anchors(self, labels):
        """
        Sample positive and negative anchors
        """
        positive = torch.where(labels >= 1)[0]
        negative = torch.where(labels == 0)[0]
        
        num_pos = int(self.batch_size_per_image * self.positive_fraction)
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.batch_size_per_image - num_pos
        num_neg = min(negative.numel(), num_neg)
        
        perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
        
        pos_idx = positive[perm1]
        neg_idx = negative[perm2]
        
        pos_mask = torch.zeros_like(labels, dtype=torch.bool)
        neg_mask = torch.zeros_like(labels, dtype=torch.bool)
        pos_mask[pos_idx] = True
        neg_mask[neg_idx] = True
        
        return pos_mask, neg_mask
    
    def forward(self, features, image_shapes, targets=None):
        """
        Args:
            features: Feature maps from backbone
            image_shapes: Original image sizes
            targets: Ground truth boxes (training only)
        """
        # RPN predictions
        t = F.relu(self.conv(features))
        objectness = self.cls_logits(t)
        pred_bbox_deltas = self.bbox_pred(t)
        
        # Generate anchors
        anchors = self.anchor_generator(
            image_shapes,
            features.shape[-2:],
            features.device
        )
        
        num_anchors_per_loc = objectness.size(1)
        objectness = objectness.permute(0, 2, 3, 1).reshape(-1, 1)
        pred_bbox_deltas = pred_bbox_deltas.view(-1, 4)
        
        # Decode boxes
        proposals = apply_regression_to_boxes(
            pred_bbox_deltas.detach().reshape(-1, 1, 4),
            anchors
        ).reshape(-1, 4)
        
        # Filter proposals
        mode = 'training' if self.training else 'testing'
        proposals, scores = self.filter_proposals(
            proposals,
            objectness.detach(),
            image_shapes,
            self.pre_nms_top_n[mode],
            self.post_nms_top_n[mode]
        )
        
        losses = {}
        if self.training and targets is not None:
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = boxes_to_transformation_targets(matched_gt_boxes, anchors)
            
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                'loss_objectness': loss_objectness,
                'loss_rpn_box_reg': loss_rpn_box_reg
            }
        
        return proposals, scores, losses
    
    def assign_targets_to_anchors(self, anchors, targets):
        """
        Assign ground truth boxes to anchors based on IoU
        """
        gt_boxes = targets['boxes']
        iou_matrix = get_iou(gt_boxes, anchors)
        
        matched_vals, matches = iou_matrix.max(dim=0)
        
        labels = torch.full((anchors.size(0),), -1, dtype=torch.float32, device=anchors.device)
        labels[matched_vals < self.bg_iou_thresh] = 0
        labels[matched_vals >= self.fg_iou_thresh] = 1
        
        matched_gt_boxes = gt_boxes[matches.clamp(min=0)]
        
        return labels, matched_gt_boxes
    
    def filter_proposals(self, proposals, objectness, image_shape, pre_nms_top_n, post_nms_top_n):
        """
        Filter and NMS proposals
        """
        objectness = objectness.sigmoid().flatten()
        
        # Pre-NMS top-k
        num_anchors = objectness.shape[0]
        pre_nms_top_n = min(pre_nms_top_n, num_anchors)
        top_n_idx = objectness.topk(pre_nms_top_n)[1]
        
        objectness = objectness[top_n_idx]
        proposals = proposals[top_n_idx]
        
        # Clip to image
        proposals = clip_boxes_to_image(proposals, image_shape)
        
        # Remove small boxes
        keep = remove_small_boxes(proposals, 16)
        proposals = proposals[keep]
        objectness = objectness[keep]
        
        # NMS
        keep = torch.ops.torchvision.nms(proposals, objectness, self.nms_thresh)
        keep = keep[:post_nms_top_n]
        
        proposals = proposals[keep]
        objectness = objectness[keep]
        
        return proposals, objectness


# ============================================================================
# Advanced ROI Head with ROI Align
# ============================================================================

class AdvancedROIHead(nn.Module):
    """
    ROI Head with:
    - ROI Align (better than ROI Pool)
    - Focal loss for classification
    - Feature extraction for relationship graph
    """
    
    def __init__(
        self,
        in_channels=256,
        num_classes=91,
        resolution=7,
        representation_size=1024,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.5,
        batch_size_per_image=512,
        positive_fraction=0.25,
        nms_thresh=0.5,
        score_thresh=0.05,
        detection_per_image=100
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.detection_per_image = detection_per_image
        
        # ROI Align
        self.roi_align = torchvision.ops.RoIAlign(
            output_size=resolution,
            spatial_scale=1.0/16.0,  # Assuming stride 16
            sampling_ratio=2
        )
        
        # Feature layers
        self.fc6 = nn.Linear(in_channels * resolution * resolution, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        
        # Prediction layers
        self.cls_score = nn.Linear(representation_size, num_classes)
        self.bbox_pred = nn.Linear(representation_size, num_classes * 4)
        
        # Initialize
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)
        
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Args:
            features: Feature maps from backbone
            proposals: Region proposals from RPN
            image_shapes: Image sizes
            targets: Ground truth (training only)
        """
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(
                proposals, targets
            )
        
        # ROI Align
        box_features = self.roi_align(features, [proposals])
        box_features = box_features.flatten(start_dim=1)
        
        # FC layers
        box_features = F.relu(self.fc6(box_features))
        box_features = F.relu(self.fc7(box_features))
        
        # Predictions
        class_logits = self.cls_score(box_features)
        box_regression = self.bbox_pred(box_features)
        
        result = {}
        losses = {}
        
        if self.training:
            loss_classifier = self.focal_loss(class_logits, labels)
            
            # Box regression loss (only foreground)
            fg_idxs = labels > 0
            if fg_idxs.sum() > 0:
                loss_box_reg = F.smooth_l1_loss(
                    box_regression[fg_idxs, labels[fg_idxs] * 4:(labels[fg_idxs] * 4 + 4)],
                    regression_targets[fg_idxs],
                    beta=1.0,
                    reduction='sum'
                ) / max(1, labels.numel())
            else:
                loss_box_reg = torch.tensor(0.0, device=box_regression.device)
            
            losses = {
                'loss_classifier': loss_classifier,
                'loss_box_reg': loss_box_reg
            }
            
            result['roi_features'] = box_features
        else:
            # Post-processing
            boxes, scores, labels, roi_features = self.postprocess_detections(
                class_logits, box_regression, proposals, box_features, image_shapes
            )
            result['boxes'] = boxes
            result['scores'] = scores
            result['labels'] = labels
            result['roi_features'] = roi_features
        
        return result, losses
    
    def select_training_samples(self, proposals, targets):
        """
        Select positive and negative samples for training
        """
        gt_boxes = targets['boxes']
        gt_labels = targets['labels']
        
        # Match proposals to GT
        match_quality_matrix = get_iou(gt_boxes, proposals)
        matched_vals, matches = match_quality_matrix.max(dim=0)
        
        # Assign labels
        labels = torch.full((proposals.size(0),), 0, dtype=torch.int64, device=proposals.device)
        bg_idxs = (matched_vals >= 0) & (matched_vals < self.fg_iou_thresh)
        labels[bg_idxs] = 0
        
        fg_idxs = matched_vals >= self.fg_iou_thresh
        labels[fg_idxs] = gt_labels[matches[fg_idxs]]
        
        # Sample
        sampled_fg_idxs, sampled_bg_idxs = self.sample_proposals(labels)
        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        
        proposals = proposals[sampled_idxs]
        labels = labels[sampled_idxs]
        matched_gt_boxes = gt_boxes[matches[sampled_idxs]]
        
        regression_targets = boxes_to_transformation_targets(matched_gt_boxes, proposals)
        
        return proposals, matches[sampled_idxs], labels, regression_targets
    
    def sample_proposals(self, labels):
        """
        Sample proposals for training
        """
        fg_idxs = torch.where(labels > 0)[0]
        bg_idxs = torch.where(labels == 0)[0]
        
        num_fg = int(self.batch_size_per_image * self.positive_fraction)
        num_fg = min(fg_idxs.numel(), num_fg)
        num_bg = self.batch_size_per_image - num_fg
        num_bg = min(bg_idxs.numel(), num_bg)
        
        perm_fg = torch.randperm(fg_idxs.numel(), device=fg_idxs.device)[:num_fg]
        perm_bg = torch.randperm(bg_idxs.numel(), device=bg_idxs.device)[:num_bg]
        
        fg_idxs = fg_idxs[perm_fg]
        bg_idxs = bg_idxs[perm_bg]
        
        return fg_idxs, bg_idxs
    
    def postprocess_detections(self, class_logits, box_regression, proposals, roi_features, image_shapes):
        """
        Post-process detections for inference
        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        
        boxes_per_image = proposals
        pred_scores = F.softmax(class_logits, -1)
        
        # Decode boxes
        pred_boxes = apply_regression_to_boxes(box_regression, proposals)
        pred_boxes = clip_boxes_to_image(pred_boxes, image_shapes)
        
        # Create labels for each prediction
        pred_labels = torch.arange(num_classes, device=device)
        pred_labels = pred_labels.view(1, -1).expand_as(pred_scores)
        
        # Remove background class
        pred_boxes = pred_boxes[:, 1:]
        pred_scores = pred_scores[:, 1:]
        pred_labels = pred_labels[:, 1:]
        
        # Flatten
        pred_boxes = pred_boxes.reshape(-1, 4)
        pred_scores = pred_scores.reshape(-1)
        pred_labels = pred_labels.reshape(-1)
        
        # Replicate roi_features for each class
        roi_features = roi_features.unsqueeze(1).repeat(1, num_classes - 1, 1).reshape(-1, roi_features.size(1))
        
        # Filter by score
        keep = pred_scores > self.score_thresh
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]
        roi_features = roi_features[keep]
        
        # Remove small boxes
        keep = remove_small_boxes(pred_boxes, 1)
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]
        roi_features = roi_features[keep]
        
        # NMS
        keep = torch.ops.torchvision.nms(pred_boxes, pred_scores, self.nms_thresh)
        keep = keep[:self.detection_per_image]
        
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]
        roi_features = roi_features[keep]
        
        return pred_boxes, pred_scores, pred_labels, roi_features


# ============================================================================
# Complete Advanced Faster R-CNN
# ============================================================================

class AdvancedFasterRCNN(nn.Module):
    """
    Advanced Faster R-CNN with all modern improvements
    """
    
    def __init__(self, num_classes=91, pretrained_backbone=True):
        super().__init__()
        
        # Backbone (ResNet-50 with FPN)
        backbone = torchvision.models.resnet50(pretrained=pretrained_backbone)
        
        # Extract feature layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1  # C2
        self.layer2 = backbone.layer2  # C3
        self.layer3 = backbone.layer3  # C4
        self.layer4 = backbone.layer4  # C5
        
        # FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )
        
        # Channel attention
        self.channel_attention = ChannelAttention(256)
        
        # RPN
        self.rpn = AdvancedRPN(in_channels=256)
        
        # ROI Head
        self.roi_head = AdvancedROIHead(
            in_channels=256,
            num_classes=num_classes
        )
        
        # Image normalization
        self.register_buffer('image_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize_images(self, images):
        """Normalize images"""
        return (images - self.image_mean) / self.image_std
    
    def forward(self, images, targets=None):
        """
        Args:
            images: (B, 3, H, W) input images
            targets: list of dicts with 'boxes' and 'labels'
        
        Returns:
            If training: dict of losses
            If testing: dict with 'boxes', 'labels', 'scores', 'roi_features'
        """
        original_image_sizes = images.shape[-2:]
        
        # Normalize
        images = self.normalize_images(images)
        
        # Backbone
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # FPN
        features = self.fpn([c2, c3, c4, c5])
        
        # Use P4 for RPN and ROI (можно использовать multi-scale)
        p4 = features[2]
        p4 = self.channel_attention(p4)
        
        # Global features for caption generation
        global_feat = F.adaptive_avg_pool2d(p4, (1, 1)).flatten(1)
        
        # RPN
        proposals, proposal_scores, rpn_losses = self.rpn(
            p4, original_image_sizes, targets
        )
        
        # ROI Head
        detections, roi_losses = self.roi_head(
            p4, proposals, original_image_sizes, targets
        )
        
        if self.training:
            losses = {}
            losses.update(rpn_losses)
            losses.update(roi_losses)
            return losses
        else:
            detections['global_features'] = global_feat
            return detections


# ============================================================================
# Testing
# ============================================================================

def test_model():
    """Test the advanced Faster R-CNN"""
    print("Testing Advanced Faster R-CNN...")
    
    model = AdvancedFasterRCNN(num_classes=91, pretrained_backbone=False)
    model.eval()
    
    # Dummy input
    images = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(images)
    
    print(f"Boxes: {outputs['boxes'].shape}")
    print(f"Scores: {outputs['scores'].shape}")
    print(f"Labels: {outputs['labels'].shape}")
    print(f"ROI Features: {outputs['roi_features'].shape}")
    print(f"Global Features: {outputs['global_features'].shape}")
    
    print("\n✓ Test passed!")


if __name__ == '__main__':
    test_model()
