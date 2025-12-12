import math
import torch
import torch.nn as nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_iou(boxes1, boxes2):
    r"""
    IOU between two sets of boxes
    :param boxes1: (Tensor of shape N x 4)
    :param boxes2: (Tensor of shape M x 4)
    :return: IOU matrix of shape N x M
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)

    x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0])   # (N, M)
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1])    # (N, M)
    x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2])  # (N, M)
    y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3]) # (N, M)

    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)
    union = area1[:, None] + area2 - intersection_area
    iou = intersection_area / union
    return iou


def boxes_to_transformation_targets(ground_truth_boxes, anchors_or_proposals):
    r"""
    Compute tx, ty, tw, th targets for anchors/proposals
    """
    widths = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    heights = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + 0.5 * widths
    center_y = anchors_or_proposals[:, 1] + 0.5 * heights

    gt_widths = ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0]
    gt_heights = ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1]
    gt_center_x = ground_truth_boxes[:, 0] + 0.5 * gt_widths
    gt_center_y = ground_truth_boxes[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_center_x - center_x) / widths
    targets_dy = (gt_center_y - center_y) / heights
    targets_dw = torch.log(gt_widths / widths)
    targets_dh = torch.log(gt_heights / heights)
    regression_targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return regression_targets


def apply_regression_pred_to_anchors_or_proposals(box_transform_pred, anchors_or_proposals):
    r"""
    Given box deltas and anchors/proposals, decode to boxes
    :param box_transform_pred: (num_anchors_or_proposals, num_classes, 4)
    :param anchors_or_proposals: (num_anchors_or_proposals, 4)
    :return pred_boxes: (num_anchors_or_proposals, num_classes, 4)
    """
    # Handle empty proposals
    if box_transform_pred.numel() == 0:
        return box_transform_pred.reshape(0, 1, 4)
    
    box_transform_pred = box_transform_pred.reshape(box_transform_pred.size(0), -1, 4)

    w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + 0.5 * w
    center_y = anchors_or_proposals[:, 1] + 0.5 * h

    dx = box_transform_pred[..., 0]
    dy = box_transform_pred[..., 1]
    dw = box_transform_pred[..., 2]
    dh = box_transform_pred[..., 3]

    dw = torch.clamp(dw, max=math.log(1000.0 / 16))
    dh = torch.clamp(dh, max=math.log(1000.0 / 16))

    # Flatten dx,dy,dw,dh if needed to match anchors
    if dx.dim() == 2:
        dx = dx.reshape(-1)
        dy = dy.reshape(-1)
        dw = dw.reshape(-1)
        dh = dh.reshape(-1)

    pred_center_x = dx * w + center_x
    pred_center_y = dy * h + center_y
    pred_w = torch.exp(dw) * w
    pred_h = torch.exp(dh) * h

    pred_box_x1 = pred_center_x - 0.5 * pred_w
    pred_box_y1 = pred_center_y - 0.5 * pred_h
    pred_box_x2 = pred_center_x + 0.5 * pred_w
    pred_box_y2 = pred_center_y + 0.5 * pred_h

    pred_boxes = torch.stack(
        (pred_box_x1, pred_box_y1, pred_box_x2, pred_box_y2), dim=1
    )
    return pred_boxes


def sample_positive_negative(labels, positive_count, total_count):
    # labels: {-1,0,1}  (ignore, bg, fg)
    positive = torch.where(labels >= 1)[0]
    negative = torch.where(labels == 0)[0]

    num_pos = min(positive.numel(), positive_count)
    num_neg = min(negative.numel(), total_count - num_pos)

    perm_positive_idxs = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm_negative_idxs = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idxs = positive[perm_positive_idxs]
    neg_idxs = negative[perm_negative_idxs]

    sampled_pos_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_neg_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_pos_idx_mask[pos_idxs] = True
    sampled_neg_idx_mask[neg_idxs] = True
    return sampled_neg_idx_mask, sampled_pos_idx_mask


def clamp_boxes_to_image_boundary(boxes, image_shape):
    boxes_x1 = boxes[..., 0]
    boxes_y1 = boxes[..., 1]
    boxes_x2 = boxes[..., 2]
    boxes_y2 = boxes[..., 3]
    height, width = image_shape[-2:]
    boxes_x1 = boxes_x1.clamp(min=0, max=width)
    boxes_x2 = boxes_x2.clamp(min=0, max=width)
    boxes_y1 = boxes_y1.clamp(min=0, max=height)
    boxes_y2 = boxes_y2.clamp(min=0, max=height)
    boxes = torch.cat(
        (
            boxes_x1[..., None],
            boxes_y1[..., None],
            boxes_x2[..., None],
            boxes_y2[..., None],
        ),
        dim=-1,
    )
    return boxes


def transform_boxes_to_original_size(boxes, new_size, original_size):
    r"""
    Convert boxes from resized image back to original size.
    """
    ratios = [
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


class RegionProposalNetwork(nn.Module):
    r"""
    RPN:
      - 3x3 conv + ReLU
      - 1x1 conv for cls (objectness)
      - 1x1 conv for bbox regression
    """

    def __init__(self, in_channels, scales, aspect_ratios, model_config):
        super().__init__()
        self.scales = scales
        self.low_iou_threshold = model_config["rpn_bg_threshold"]
        self.high_iou_threshold = model_config["rpn_fg_threshold"]
        self.rpn_nms_threshold = model_config["rpn_nms_threshold"]
        self.rpn_batch_size = model_config["rpn_batch_size"]
        self.rpn_pos_count = int(model_config["rpn_pos_fraction"] * self.rpn_batch_size)

        # Lưu riêng train/test topk (fix bug self.training trong __init__)
        self.rpn_train_topk = model_config["rpn_train_topk"]
        self.rpn_test_topk = model_config["rpn_test_topk"]
        self.rpn_train_prenms_topk = model_config["rpn_train_prenms_topk"]
        self.rpn_test_prenms_topk = model_config["rpn_test_prenms_topk"]

        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(self.scales) * len(self.aspect_ratios)

        self.rpn_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_layer = nn.Conv2d(in_channels, self.num_anchors, kernel_size=1, stride=1)
        self.bbox_reg_layer = nn.Conv2d(in_channels, self.num_anchors * 4, kernel_size=1, stride=1)

        for layer in [self.rpn_conv, self.cls_layer, self.bbox_reg_layer]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def generate_anchors(self, image, feat):
        r"""
        Generate anchors for entire feature map
        """
        grid_h, grid_w = feat.shape[-2:]
        image_h, image_w = image.shape[-2:]

        stride_h = torch.tensor(image_h // grid_h, dtype=torch.int64, device=feat.device)
        stride_w = torch.tensor(image_w // grid_w, dtype=torch.int64, device=feat.device)

        scales = torch.as_tensor(self.scales, dtype=feat.dtype, device=feat.device)
        aspect_ratios = torch.as_tensor(self.aspect_ratios, dtype=feat.dtype, device=feat.device)

        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        base_anchors = base_anchors.round()

        shifts_x = torch.arange(0, grid_w, dtype=torch.int32, device=feat.device) * stride_w
        shifts_y = torch.arange(0, grid_h, dtype=torch.int32, device=feat.device) * stride_h

        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shifts_x = shifts_x.reshape(-1)
        shifts_y = shifts_y.reshape(-1)
        shifts = torch.stack((shifts_x, shifts_y, shifts_x, shifts_y), dim=1)

        anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4))
        anchors = anchors.reshape(-1, 4)
        return anchors

    def assign_targets_to_anchors(self, anchors, gt_boxes):
        r"""
        IOU-based matching: assign each anchor to gt box / background / ignore
        """
        iou_matrix = get_iou(gt_boxes, anchors)

        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)
        best_match_gt_idx_pre_thresholding = best_match_gt_idx.clone()

        below_low_threshold = best_match_iou < self.low_iou_threshold
        between_thresholds = (best_match_iou >= self.low_iou_threshold) & (
            best_match_iou < self.high_iou_threshold
        )
        best_match_gt_idx[below_low_threshold] = -1
        best_match_gt_idx[between_thresholds] = -2

        best_anchor_iou_for_gt, _ = iou_matrix.max(dim=1)
        gt_pred_pair_with_highest_iou = torch.where(iou_matrix == best_anchor_iou_for_gt[:, None])
        pred_inds_to_update = gt_pred_pair_with_highest_iou[1]
        best_match_gt_idx[pred_inds_to_update] = best_match_gt_idx_pre_thresholding[pred_inds_to_update]

        matched_gt_boxes = gt_boxes[best_match_gt_idx.clamp(min=0)]

        labels = best_match_gt_idx >= 0
        labels = labels.to(dtype=torch.float32)
        background_anchors = best_match_gt_idx == -1
        labels[background_anchors] = 0.0
        ignored_anchors = best_match_gt_idx == -2
        labels[ignored_anchors] = -1.0
        return labels, matched_gt_boxes

    def filter_proposals(self, proposals, cls_scores, image_shape, prenms_topk, topk):
        r"""
        1. Pre-NMS topK
        2. Clamp to image
        3. Remove small boxes
        4. NMS
        5. Post-NMS topK
        """
        cls_scores = cls_scores.reshape(-1)
        cls_scores = torch.sigmoid(cls_scores)
        _, top_n_idx = cls_scores.topk(min(prenms_topk, len(cls_scores)))

        cls_scores = cls_scores[top_n_idx]
        proposals = proposals[top_n_idx]

        proposals = clamp_boxes_to_image_boundary(proposals, image_shape)

        min_size = 16
        ws, hs = proposals[:, 2] - proposals[:, 0], proposals[:, 3] - proposals[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        proposals = proposals[keep]
        cls_scores = cls_scores[keep]

        keep_mask = torch.zeros_like(cls_scores, dtype=torch.bool)
        keep_indices = torch.ops.torchvision.nms(proposals, cls_scores, self.rpn_nms_threshold)
        keep_mask[keep_indices] = True
        keep_indices = torch.where(keep_mask)[0]
        post_nms_keep_indices = keep_indices[cls_scores[keep_indices].sort(descending=True)[1]]

        proposals = proposals[post_nms_keep_indices[:topk]]
        cls_scores = cls_scores[post_nms_keep_indices[:topk]]
        return proposals, cls_scores

    def forward(self, image, feat, target=None):
        r"""
        Main RPN forward - process each image in batch separately
        """
        batch_size = feat.size(0)
        rpn_feat = nn.ReLU()(self.rpn_conv(feat))
        cls_scores = self.cls_layer(rpn_feat)
        box_transform_pred = self.bbox_reg_layer(rpn_feat)

        # Generate anchors once (same for all images)
        anchors = self.generate_anchors(image[0:1], feat[0:1])
        num_anchors = anchors.size(0)

        number_of_anchors_per_location = cls_scores.size(1)
        
        # Process each image separately
        all_proposals = []
        all_scores = []
        all_cls_losses = []
        all_loc_losses = []

        for idx in range(batch_size):
            # Get predictions for this image
            img_cls_scores = cls_scores[idx:idx+1]
            img_box_pred = box_transform_pred[idx:idx+1]
            
            img_cls_scores = img_cls_scores.permute(0, 2, 3, 1).reshape(-1, 1)
            
            img_box_pred = img_box_pred.view(
                1, number_of_anchors_per_location, 4,
                rpn_feat.shape[-2], rpn_feat.shape[-1]
            )
            img_box_pred = img_box_pred.permute(0, 3, 4, 1, 2).reshape(-1, 4)

            # Apply regression to get proposals
            img_proposals = apply_regression_pred_to_anchors_or_proposals(
                img_box_pred.detach(), anchors
            )

            # Filter proposals
            if self.training:
                prenms_topk = self.rpn_train_prenms_topk
                topk = self.rpn_train_topk
            else:
                prenms_topk = self.rpn_test_prenms_topk
                topk = self.rpn_test_topk

            img_proposals, img_scores = self.filter_proposals(
                img_proposals, img_cls_scores.detach(), 
                image[idx:idx+1].shape, prenms_topk, topk
            )
            
            all_proposals.append(img_proposals)
            all_scores.append(img_scores)

            # Compute losses if training
            if self.training and target is not None:
                gt_boxes = target["bboxes"][idx]
                labels_for_anchors, matched_gt_boxes = self.assign_targets_to_anchors(
                    anchors, gt_boxes
                )
                regression_targets = boxes_to_transformation_targets(
                    matched_gt_boxes, anchors
                )

                sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(
                    labels_for_anchors,
                    positive_count=self.rpn_pos_count,
                    total_count=self.rpn_batch_size,
                )

                sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]

                if sampled_pos_idx_mask.sum() > 0:
                    loc_loss = nn.functional.smooth_l1_loss(
                        img_box_pred[sampled_pos_idx_mask],
                        regression_targets[sampled_pos_idx_mask],
                        beta=1 / 9,
                        reduction="sum",
                    ) / sampled_idxs.numel()
                else:
                    loc_loss = torch.tensor(0.0, device=feat.device)

                cls_loss = nn.functional.binary_cross_entropy_with_logits(
                    img_cls_scores[sampled_idxs].flatten(),
                    labels_for_anchors[sampled_idxs].flatten(),
                )
                
                all_cls_losses.append(cls_loss)
                all_loc_losses.append(loc_loss)

        # Combine results
        rpn_output = {
            "proposals": all_proposals,
            "scores": all_scores
        }

        if self.training and target is not None:
            rpn_output["rpn_classification_loss"] = torch.stack(all_cls_losses).mean()
            rpn_output["rpn_localization_loss"] = torch.stack(all_loc_losses).mean()

        return rpn_output


class ROIHead(nn.Module):
    r"""
    ROI head:
      - roi_pool
      - fc6, fc7
      - cls & bbox layer
    Ngoài boxes/scores/labels, mình thêm:
      - roi_features: đặc trưng vùng (fc7) cho từng detection
    """

    def __init__(self, model_config, num_classes, in_channels):
        super().__init__()
        self.num_classes = num_classes
        self.roi_batch_size = model_config["roi_batch_size"]
        self.roi_pos_count = int(model_config["roi_pos_fraction"] * self.roi_batch_size)
        self.iou_threshold = model_config["roi_iou_threshold"]
        self.low_bg_iou = model_config["roi_low_bg_iou"]
        self.nms_threshold = model_config["roi_nms_threshold"]
        self.topK_detections = model_config["roi_topk_detections"]
        self.low_score_threshold = model_config["roi_score_threshold"]
        self.pool_size = model_config["roi_pool_size"]
        self.fc_inner_dim = model_config["fc_inner_dim"]

        self.fc6 = nn.Linear(in_channels * self.pool_size * self.pool_size, self.fc_inner_dim)
        self.fc7 = nn.Linear(self.fc_inner_dim, self.fc_inner_dim)
        self.cls_layer = nn.Linear(self.fc_inner_dim, self.num_classes)
        self.bbox_reg_layer = nn.Linear(self.fc_inner_dim, self.num_classes * 4)

        nn.init.normal_(self.cls_layer.weight, std=0.01)
        nn.init.constant_(self.cls_layer.bias, 0)
        nn.init.normal_(self.bbox_reg_layer.weight, std=0.001)
        nn.init.constant_(self.bbox_reg_layer.bias, 0)

    def assign_target_to_proposals(self, proposals, gt_boxes, gt_labels):
        r"""
        Assign each proposal to gt box / background / ignore
        """
        iou_matrix = get_iou(gt_boxes, proposals)
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)

        background_proposals = (best_match_iou < self.iou_threshold) & (
            best_match_iou >= self.low_bg_iou
        )
        ignored_proposals = best_match_iou < self.low_bg_iou

        best_match_gt_idx[background_proposals] = -1
        best_match_gt_idx[ignored_proposals] = -2

        matched_gt_boxes_for_proposals = gt_boxes[best_match_gt_idx.clamp(min=0)]

        labels = gt_labels[best_match_gt_idx.clamp(min=0)].to(dtype=torch.int64)
        labels[background_proposals] = 0
        labels[ignored_proposals] = -1
        return labels, matched_gt_boxes_for_proposals

    def forward(self, feat, proposals_list, image_shape, target):
        """
        Process each image in batch separately
        proposals_list: list of proposals for each image
        """
        batch_size = feat.size(0)
        all_outputs = []
        
        for idx in range(batch_size):
            img_feat = feat[idx:idx+1]
            proposals = proposals_list[idx]
            
            if self.training and target is not None:
                gt_boxes = target["bboxes"][idx]
                gt_labels = target["labels"][idx]
                
                proposals = torch.cat([proposals, gt_boxes], dim=0)

                labels, matched_gt_boxes = self.assign_target_to_proposals(
                    proposals, gt_boxes, gt_labels
                )

                sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(
                    labels,
                    positive_count=self.roi_pos_count,
                    total_count=self.roi_batch_size,
                )
                sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]

                proposals = proposals[sampled_idxs]
                labels = labels[sampled_idxs]
                matched_gt_boxes = matched_gt_boxes[sampled_idxs]
                regression_targets = boxes_to_transformation_targets(
                    matched_gt_boxes, proposals
                )
            
            # Feature extraction
            size = img_feat.shape[-2:]
            possible_scales = []
            for s1, s2 in zip(size, image_shape):
                approx_scale = float(s1) / float(s2)
                scale = 2 ** float(torch.tensor(approx_scale).log2().round())
                possible_scales.append(scale)
            assert possible_scales[0] == possible_scales[1]

            proposal_roi_pool_feats = torchvision.ops.roi_pool(
                img_feat,
                [proposals],
                output_size=self.pool_size,
                spatial_scale=possible_scales[0],
            )
            proposal_roi_pool_feats = proposal_roi_pool_feats.flatten(start_dim=1)
            box_fc_6 = nn.functional.relu(self.fc6(proposal_roi_pool_feats))
            box_fc_7 = nn.functional.relu(self.fc7(box_fc_6))
            cls_scores = self.cls_layer(box_fc_7)
            box_transform_pred = self.bbox_reg_layer(box_fc_7)

            num_boxes, num_classes = cls_scores.shape
            box_transform_pred = box_transform_pred.reshape(num_boxes, num_classes, 4)

            frcnn_output = {}

            if self.training and target is not None:
                classification_loss = nn.functional.cross_entropy(cls_scores, labels)

                fg_proposals_idxs = torch.where(labels > 0)[0]
                if len(fg_proposals_idxs) > 0:
                    fg_cls_labels = labels[fg_proposals_idxs]

                    localization_loss = nn.functional.smooth_l1_loss(
                        box_transform_pred[fg_proposals_idxs, fg_cls_labels],
                        regression_targets[fg_proposals_idxs],
                        beta=1 / 9,
                        reduction="sum",
                    ) / labels.numel()
                else:
                    localization_loss = torch.tensor(0.0, device=feat.device)

                frcnn_output["frcnn_classification_loss"] = classification_loss
                frcnn_output["frcnn_localization_loss"] = localization_loss
                frcnn_output["roi_features"] = box_fc_7
            else:
                device = cls_scores.device
                
                # box_transform_pred: (num_proposals, num_classes, 4)
                # proposals: (num_proposals, 4)
                # Need to expand proposals for each class
                proposals_expanded = proposals.unsqueeze(1).expand(-1, num_classes, -1).reshape(-1, 4)
                box_transform_pred_flat = box_transform_pred.reshape(-1, 4)
                
                pred_boxes = apply_regression_pred_to_anchors_or_proposals(
                    box_transform_pred_flat, proposals_expanded
                )
                # Reshape back to (num_proposals, num_classes, 4)
                pred_boxes = pred_boxes.reshape(num_boxes, num_classes, 4)
                
                pred_scores = nn.functional.softmax(cls_scores, dim=-1)

                pred_boxes = clamp_boxes_to_image_boundary(pred_boxes.reshape(-1, 4), image_shape)
                pred_boxes = pred_boxes.reshape(num_boxes, num_classes, 4)

                pred_labels = torch.arange(num_classes, device=device)
                pred_labels = pred_labels.view(1, -1).expand_as(pred_scores)

                pred_boxes = pred_boxes[:, 1:]
                pred_scores = pred_scores[:, 1:]
                pred_labels = pred_labels[:, 1:]

                pred_boxes = pred_boxes.reshape(-1, 4)
                pred_scores = pred_scores.reshape(-1)
                pred_labels = pred_labels.reshape(-1)

                num_classes_no_bg = num_classes - 1
                roi_feats_flat = (
                    box_fc_7.unsqueeze(1)
                    .expand(-1, num_classes_no_bg, -1)
                    .reshape(-1, box_fc_7.size(1))
                )

                pred_boxes, pred_labels, pred_scores, roi_feats_flat = self.filter_predictions(
                    pred_boxes, pred_labels, pred_scores, roi_feats_flat
                )
                frcnn_output["boxes"] = pred_boxes
                frcnn_output["scores"] = pred_scores
                frcnn_output["labels"] = pred_labels
                frcnn_output["roi_features"] = roi_feats_flat
            
            all_outputs.append(frcnn_output)
        
        # Combine batch outputs
        if self.training and target is not None:
            combined_output = {
                "frcnn_classification_loss": torch.stack([o["frcnn_classification_loss"] for o in all_outputs]).mean(),
                "frcnn_localization_loss": torch.stack([o["frcnn_localization_loss"] for o in all_outputs]).mean(),
                "roi_features": [o["roi_features"] for o in all_outputs]
            }
        else:
            combined_output = {
                "boxes": [o["boxes"] for o in all_outputs],
                "scores": [o["scores"] for o in all_outputs],
                "labels": [o["labels"] for o in all_outputs],
                "roi_features": [o["roi_features"] for o in all_outputs]
            }
        
        return combined_output

    def filter_predictions(self, pred_boxes, pred_labels, pred_scores, pred_feats):
        r"""
        Lọc prediction:
          1. bỏ low-score
          2. bỏ box nhỏ
          3. NMS theo từng class
          4. giữ topK
        Đồng thời apply y chang cho pred_feats (roi features)
        """
        # low score filter
        keep = torch.where(pred_scores > self.low_score_threshold)[0]
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]
        pred_feats = pred_feats[keep]

        # remove small boxes
        min_size = 16
        ws, hs = pred_boxes[:, 2] - pred_boxes[:, 0], pred_boxes[:, 3] - pred_boxes[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]
        pred_feats = pred_feats[keep]

        # class-wise NMS
        keep_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
        for class_id in torch.unique(pred_labels):
            curr_indices = torch.where(pred_labels == class_id)[0]
            curr_keep_indices = torch.ops.torchvision.nms(
                pred_boxes[curr_indices],
                pred_scores[curr_indices],
                self.nms_threshold,
            )
            keep_mask[curr_indices[curr_keep_indices]] = True
        keep_indices = torch.where(keep_mask)[0]
        post_nms_keep_indices = keep_indices[pred_scores[keep_indices].sort(descending=True)[1]]
        keep = post_nms_keep_indices[: self.topK_detections]

        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]
        pred_feats = pred_feats[keep]
        return pred_boxes, pred_labels, pred_scores, pred_feats


class FasterRCNN(nn.Module):
    def __init__(self, model_config, num_classes):
        super().__init__()
        self.model_config = model_config

        # Không dùng pretrained để đúng kiểu "tự train"
        vgg16 = torchvision.models.vgg16(pretrained=False)
        self.backbone = vgg16.features[:-1]

        self.rpn = RegionProposalNetwork(
            model_config["backbone_out_channels"],
            scales=model_config["scales"],
            aspect_ratios=model_config["aspect_ratios"],
            model_config=model_config,
        )
        self.roi_head = ROIHead(
            model_config,
            num_classes,
            in_channels=model_config["backbone_out_channels"],
        )

        # freeze sớm vài layer backbone
        for layer in self.backbone[:10]:
            for p in layer.parameters():
                p.requires_grad = False

        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.min_size = model_config["min_im_size"]
        self.max_size = model_config["max_im_size"]

    def normalize_resize_image_and_boxes(self, image, bboxes):
        dtype, device = image.dtype, image.device

        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        image = (image - mean[:, None, None]) / std[:, None, None]

        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(float(self.min_size) / min_size, float(self.max_size) / max_size)
        scale_factor = scale.item()

        image = nn.functional.interpolate(
            image,
            size=None,
            scale_factor=scale_factor,
            mode="bilinear",
            recompute_scale_factor=True,
            align_corners=False,
        )

        if bboxes is not None:
            ratios = [
                torch.tensor(s, dtype=torch.float32, device=bboxes.device)
                / torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
                for s, s_orig in zip(image.shape[-2:], (h, w))
            ]
            ratio_height, ratio_width = ratios
            xmin, ymin, xmax, ymax = bboxes.unbind(2)
            xmin = xmin * ratio_width
            xmax = xmax * ratio_width
            ymin = ymin * ratio_height
            ymax = ymax * ratio_height
            bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=2)
        return image, bboxes

    def forward(self, image, targets=None):
        """
        image: (B, C, H, W)
        targets: dict with 'bboxes' (B, N, 4) and 'labels' (B, N)
        """
        old_shape = image.shape[-2:]
        batch_size = image.shape[0]

        # Process batch: normalize/resize all images
        if self.training and targets is not None:
            # Convert targets from dict format to list format for batch processing
            targets_list = []
            images_norm = []
            
            for i in range(batch_size):
                img, boxes = self.normalize_resize_image_and_boxes(
                    image[i:i+1], 
                    targets["bboxes"][i:i+1]
                )
                images_norm.append(img)
                targets_list.append({
                    'bboxes': boxes.squeeze(0),
                    'labels': targets['labels'][i]
                })
            image = torch.cat(images_norm, dim=0)
            targets = {'bboxes': [t['bboxes'] for t in targets_list],
                      'labels': [t['labels'] for t in targets_list]}
        else:
            # Normalize images (no boxes)
            images_norm = []
            for i in range(batch_size):
                img, _ = self.normalize_resize_image_and_boxes(image[i:i+1], None)
                images_norm.append(img)
            image = torch.cat(images_norm, dim=0)

        feat = self.backbone(image)

        # Global feature for caption
        global_feat = nn.functional.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)  # (B, C)

        rpn_output = self.rpn(image, feat, targets)
        proposals = rpn_output["proposals"]

        frcnn_output = self.roi_head(feat, proposals, image.shape[-2:], targets)

        if not self.training:
            # Transform boxes back for each image
            boxes_original = []
            for boxes in frcnn_output["boxes"]:
                boxes_transformed = transform_boxes_to_original_size(
                    boxes, image.shape[-2:], old_shape
                )
                boxes_original.append(boxes_transformed)
            frcnn_output["boxes"] = boxes_original
        
        frcnn_output["global_feat"] = global_feat  # (B, C) – dùng làm context cho LSTM

        return rpn_output, frcnn_output
