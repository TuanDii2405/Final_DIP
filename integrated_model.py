"""
Combined Model: Faster R-CNN + Relationship Graph + LSTM Caption Generator
Integrated pipeline for image captioning with visual relationships
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

# Import components
from faster_rcnn import FasterRCNN
from relationship_graph_veto import VisualRelationshipGraph
from caption_generator import CaptionLSTM, Vocabulary


class IntegratedCaptionModel(nn.Module):
    """
    Complete model integrating:
    1. Faster R-CNN for object detection
    2. Relationship Graph for visual relationships
    3. LSTM for caption generation
    
    Features are combined at multiple levels for better captioning
    """
    
    def __init__(
        self,
        num_object_classes=91,
        vocab_size=10000,
        embedding_dim=512,
        hidden_dim=512,
        num_relationship_layers=3,
        num_spatial_rels=8,
        num_semantic_rels=50,
        dropout=0.1,
        pretrained_backbone=True
    ):
        super().__init__()
        
        self.num_object_classes = num_object_classes
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # ========== Component 1: Faster R-CNN ==========
        # Create model config for Faster R-CNN
        model_config = {
            'backbone_out_channels': 512,
            'scales': [128, 256, 512],
            'aspect_ratios': [0.5, 1.0, 2.0],
            'rpn_bg_threshold': 0.3,
            'rpn_fg_threshold': 0.7,
            'rpn_nms_threshold': 0.7,
            'rpn_batch_size': 256,
            'rpn_pos_fraction': 0.5,
            'rpn_train_topk': 2000,
            'rpn_test_topk': 1000,
            'rpn_train_prenms_topk': 2000,
            'rpn_test_prenms_topk': 1000,
            'roi_batch_size': 128,
            'roi_pos_fraction': 0.25,
            'roi_iou_threshold': 0.5,
            'roi_low_bg_iou': 0.0,
            'roi_nms_threshold': 0.5,
            'roi_topk_detections': 100,
            'roi_score_threshold': 0.05,
            'roi_pool_size': 7,
            'fc_inner_dim': 1024,
            'min_im_size': 600,
            'max_im_size': 1000
        }
        
        self.faster_rcnn = FasterRCNN(
            model_config=model_config,
            num_classes=num_object_classes
        )
        
        # ========== Component 2: Relationship Graph ==========
        # ROI features từ Faster R-CNN có dim 1024, cần project về 512
        self.roi_feature_proj = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.relationship_graph = VisualRelationshipGraph(
            object_feature_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=8,
            num_layers=num_relationship_layers,
            num_spatial_rels=num_spatial_rels,
            num_semantic_rels=num_semantic_rels,
            dropout=dropout
        )
        
        # ========== Component 3: Feature Fusion ==========
        # Combine object features, relationship features, and global features
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 256, hidden_dim),  # obj + rel + global
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Project global features (Faster R-CNN gốc output 512 dim)
        self.global_feature_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(dropout)
        )
        
        # ========== Component 4: LSTM Caption Generator ==========
        self.caption_generator = CaptionLSTM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout
        )
        
        # ========== Auxiliary Heads ==========
        # Multi-task learning: помогает train detection + relationship
        self.object_classifier = nn.Linear(hidden_dim, num_object_classes)
        
        # Attention mechanism for selecting important objects
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(
        self,
        images,
        captions=None,
        caption_lengths=None,
        targets=None,
        mode='train'
    ):
        """
        Forward pass through entire pipeline
        
        Args:
            images: (B, 3, H, W) input images
            captions: (B, max_len) caption tokens (training only)
            caption_lengths: (B,) caption lengths (training only)
            targets: dict with 'boxes' and 'labels' (training only)
            mode: 'train', 'inference', or 'features'
        
        Returns:
            If training: dict of losses
            If inference: dict with generated captions + detections
            If features: dict with all intermediate features
        """
        batch_size = images.size(0)
        device = images.device
        
        # ========== Step 1: Object Detection ==========
        device = images.device
        batch_size = images.size(0)
        assert batch_size == 1, "Current implementation only supports batch_size=1"
        
        if self.training and mode == 'train':
            # Training mode: get losses from Faster R-CNN
            rpn_out, frcnn_out = self.faster_rcnn(images, targets)
            
            # Extract losses
            frcnn_losses = {}
            if 'rpn_classification_loss' in rpn_out:
                frcnn_losses['loss_objectness'] = rpn_out['rpn_classification_loss']
            if 'rpn_localization_loss' in rpn_out:
                frcnn_losses['loss_rpn_box_reg'] = rpn_out['rpn_localization_loss']
            if 'frcnn_classification_loss' in frcnn_out:
                frcnn_losses['loss_classifier'] = frcnn_out['frcnn_classification_loss']
            if 'frcnn_localization_loss' in frcnn_out:
                frcnn_losses['loss_box_reg'] = frcnn_out['frcnn_localization_loss']
            
            # Get features from training output (roi_features is list of tensors)
            roi_features_list = frcnn_out.get('roi_features', [torch.zeros(0, 1024, device=device)])
            global_features = frcnn_out.get('global_feat', torch.zeros(1, 512, device=device))
        else:
            # Inference mode
            rpn_out, frcnn_out = self.faster_rcnn(images, None)
            frcnn_losses = {}
            
            roi_features_list = frcnn_out.get('roi_features', [torch.zeros(0, 1024, device=device)])
            boxes_list = frcnn_out.get('boxes', [torch.zeros(0, 4, device=device)])
            global_features = frcnn_out.get('global_feat', torch.zeros(1, 512, device=device))
        
        # Extract features for the single image (batch=1)
        roi_features = roi_features_list[0]  # (N, 1024)
        global_feat = global_features[0:1]  # (1, 512)
        
        # Handle case when no objects detected
        if roi_features.size(0) == 0:
            # Use global features only
            global_proj = self.global_feature_proj(global_feat)  # (1, 256)
            context = global_proj.unsqueeze(1)  # (1, 1, 256)
            context = F.pad(context, (0, self.hidden_dim - 256))  # (1, 1, 512)
            
            if mode == 'train':
                caption_loss = self.caption_generator.compute_loss(
                    context, captions, caption_lengths
                )
                total_loss = caption_loss
                for k, v in frcnn_losses.items():
                    total_loss = total_loss + v
                
                return {
                    'total_loss': total_loss,
                    'caption_loss': caption_loss,
                    **frcnn_losses
                }
            else:
                generated_captions = self.caption_generator.generate(
                    context, max_length=20
                )
                return {
                    'captions': generated_captions,
                    'boxes': boxes_list[0] if not self.training else torch.zeros(0, 4, device=device),
                    'num_objects': 0
                }
        
        # ========== Step 2: Project ROI Features ==========
        obj_features = self.roi_feature_proj(roi_features)  # (N, 256)
        
        # Limit to top objects
        num_objects = min(obj_features.size(0), 10)
        obj_features = obj_features[:num_objects]
        
        # Get boxes for relationship graph
        if not self.training or mode != 'train':
            boxes = boxes_list[0][:num_objects]  # (N, 4)
        else:
            # In training mode, create dummy boxes (relationship graph needs them)
            boxes = torch.zeros(num_objects, 4, device=device)
        
        # ========== Step 3: Relationship Graph ==========
        obj_features_batch = obj_features.unsqueeze(0)  # (1, N, 256)
        boxes_batch = boxes.unsqueeze(0)  # (1, N, 4)
        
        rel_features, relationship_preds = self.relationship_graph(
            obj_features_batch,
            boxes_batch,
            return_relationships=True
        )  # (1, N, 256)
        
        rel_features = rel_features.squeeze(0)  # (N, 256)
        
        # ========== Step 4: Feature Fusion ==========
        global_proj = self.global_feature_proj(global_feat)  # (1, 256)
        global_expanded = global_proj.expand(num_objects, -1)  # (N, 256)
        
        # Concatenate all features
        combined_features = torch.cat([
            obj_features,      # (N, 256)
            rel_features,      # (N, 256)
            global_expanded    # (N, 256)
        ], dim=-1)  # (N, 768)
        
        fused_features = self.feature_fusion(combined_features)  # (N, 512)
        
        # ========== Step 5: Attention over Objects ==========
        fused_features_batch = fused_features.unsqueeze(0)  # (1, N, 512)
        
        attn_output, attn_weights = self.attention(
            fused_features_batch,
            fused_features_batch,
            fused_features_batch
        )  # (1, N, 512)
        
        # ========== Step 6: Generate Caption ==========
        context = attn_output  # (1, N, 512)
        
        if mode == 'train' and captions is not None:
            # Training: compute caption loss
            caption_loss = self.caption_generator.compute_loss(
                context, captions, caption_lengths
            )
            
            # Total loss
            total_loss = caption_loss
            
            # Add Faster R-CNN losses
            for k, v in frcnn_losses.items():
                total_loss = total_loss + v
            
            return {
                'total_loss': total_loss,
                'caption_loss': caption_loss,
                **frcnn_losses
            }
        else:
            # Inference: generate caption
            generated_captions = self.caption_generator.generate(
                context, max_length=20
            )
            
            return {
                'captions': generated_captions,
                'boxes': boxes,
                'num_objects': num_objects
            }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(vocab_size, config):
    """Build the integrated caption model"""
    model = IntegratedCaptionModel(
        vocab_size=vocab_size,
        embedding_dim=config.get('embedding_dim', 512),
        hidden_dim=config['hidden_dim'],
        num_relationship_layers=config['num_relationship_layers'],
        num_spatial_rels=config['num_spatial_rels'],
        num_semantic_rels=config['num_semantic_rels']
    )
    return model


class CombinedLoss(nn.Module):
    """Combined loss for training"""
    def __init__(self):
        super().__init__()
    
    def forward(self, outputs):
        """Simply return the total_loss from model outputs"""
        return outputs['total_loss']
