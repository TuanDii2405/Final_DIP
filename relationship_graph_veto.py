"""
Visual Relationship Graph Module
Inspired by VeTO (Vision Transformer for Dense Prediction)
https://github.com/visinf/veto

Architecture:
- Object Node Features (from Faster R-CNN)
- Spatial Features (geometric relationships)
- Visual Features (appearance similarity)
- Multi-head Graph Attention
- Message Passing for relationship reasoning
- Semantic & Spatial Relationship Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for spatial coordinates"""
    
    def __init__(self, d_model=256, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class SpatialFeatureExtractor(nn.Module):
    """
    Extract geometric relationship features between object pairs
    Features include:
    - Relative position (dx, dy, dw, dh)
    - Distance and angle
    - Overlap (IoU)
    - Aspect ratio
    """
    
    def __init__(self, input_dim=11, hidden_dim=256, output_dim=512):
        super().__init__()
        self.input_dim = input_dim
        
        # MLP to encode spatial features
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim)
        )
    
    def compute_spatial_features(self, boxes1, boxes2):
        """
        Compute spatial features between two sets of boxes
        
        Args:
            boxes1: (N, 4) tensor [x1, y1, x2, y2]
            boxes2: (M, 4) tensor [x1, y1, x2, y2]
        
        Returns:
            features: (N, M, input_dim) tensor of spatial features
        """
        N = boxes1.size(0)
        M = boxes2.size(0)
        
        # Compute centers, widths, heights
        x1_c = (boxes1[:, 0] + boxes1[:, 2]) / 2  # (N,)
        y1_c = (boxes1[:, 1] + boxes1[:, 3]) / 2
        w1 = boxes1[:, 2] - boxes1[:, 0]
        h1 = boxes1[:, 3] - boxes1[:, 1]
        
        x2_c = (boxes2[:, 0] + boxes2[:, 2]) / 2  # (M,)
        y2_c = (boxes2[:, 1] + boxes2[:, 3]) / 2
        w2 = boxes2[:, 2] - boxes2[:, 0]
        h2 = boxes2[:, 3] - boxes2[:, 1]
        
        # Pairwise features (N, M)
        dx = (x2_c[None, :] - x1_c[:, None]) / (w1[:, None] + 1e-5)  # Relative x distance
        dy = (y2_c[None, :] - y1_c[:, None]) / (h1[:, None] + 1e-5)  # Relative y distance
        dw = torch.log((w2[None, :] + 1e-5) / (w1[:, None] + 1e-5))  # Log width ratio
        dh = torch.log((h2[None, :] + 1e-5) / (h1[:, None] + 1e-5))  # Log height ratio
        
        # Euclidean distance
        distance = torch.sqrt(dx**2 + dy**2 + 1e-8)
        
        # Angle
        angle = torch.atan2(dy, dx)
        
        # IoU (Intersection over Union)
        iou = self.compute_iou(boxes1, boxes2)
        
        # Aspect ratios
        aspect1 = w1 / (h1 + 1e-5)
        aspect2 = w2 / (h2 + 1e-5)
        aspect_ratio = (aspect2[None, :] / (aspect1[:, None] + 1e-5))
        
        # Area ratios
        area1 = w1 * h1
        area2 = w2 * h2
        area_ratio = (area2[None, :] / (area1[:, None] + 1e-5))
        
        # Stack all features: (N, M, 11)
        features = torch.stack([
            dx, dy, dw, dh,           # 4: relative position & scale
            distance, angle,           # 2: distance and direction
            iou,                       # 1: overlap
            aspect_ratio, area_ratio,  # 2: shape similarity
            torch.sin(angle),          # 1: directional encoding
            torch.cos(angle)           # 1: directional encoding
        ], dim=2)
        
        return features
    
    def compute_iou(self, boxes1, boxes2):
        """Compute IoU between all pairs of boxes"""
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
    
    def forward(self, boxes):
        """
        Args:
            boxes: (N, 4) bounding boxes
        
        Returns:
            spatial_features: (N, N, output_dim) pairwise spatial features
        """
        spatial_raw = self.compute_spatial_features(boxes, boxes)  # (N, N, 11)
        N = boxes.size(0)
        
        # Flatten for MLP
        spatial_flat = spatial_raw.view(-1, self.input_dim)  # (N*N, 11)
        encoded = self.spatial_encoder(spatial_flat)  # (N*N, output_dim)
        spatial_features = encoded.view(N, N, -1)  # (N, N, output_dim)
        
        return spatial_features


class MultiHeadGraphAttention(nn.Module):
    """
    Multi-head attention mechanism for graph nodes
    Similar to Transformer but adapted for relationship graphs
    """
    
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, query, key, value, edge_features=None, mask=None):
        """
        Args:
            query: (batch, N, d_model)
            key: (batch, N, d_model)
            value: (batch, N, d_model)
            edge_features: (batch, N, N, d_model) optional edge features
            mask: (batch, N, N) attention mask
        
        Returns:
            output: (batch, N, d_model)
            attention_weights: (batch, num_heads, N, N)
        """
        batch_size = query.size(0)
        N = query.size(1)
        
        # Linear projections and split heads
        Q = self.W_q(query).view(batch_size, N, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, N, d_k)
        K = self.W_k(key).view(batch_size, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, N, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, N, N)
        
        # Add edge features if provided
        if edge_features is not None:
            # Project edge features to match num_heads
            edge_bias = edge_features.mean(dim=-1, keepdim=True)  # (B, N, N, 1)
            edge_bias = edge_bias.squeeze(-1).unsqueeze(1)  # (B, 1, N, N)
            scores = scores + edge_bias
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)  # (B, H, N, N)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # (B, H, N, d_k)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, N, self.d_model)
        
        # Output projection
        output = self.W_o(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(query + self.dropout(output))
        
        return output, attention_weights


class GraphConvolutionalLayer(nn.Module):
    """
    Graph Convolutional Layer with message passing
    """
    
    def __init__(self, in_features=512, out_features=512, dropout=0.1):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, node_features, adjacency_matrix):
        """
        Args:
            node_features: (batch, N, in_features)
            adjacency_matrix: (batch, N, N) or (N, N)
        
        Returns:
            updated_features: (batch, N, out_features)
        """
        # Message passing: aggregate neighbor features
        if adjacency_matrix.dim() == 2:
            adjacency_matrix = adjacency_matrix.unsqueeze(0)
        
        # Normalize adjacency matrix (add self-loops and compute degree)
        adj = adjacency_matrix + torch.eye(adjacency_matrix.size(-1), device=adjacency_matrix.device)
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj_norm = adj / degree
        
        # Aggregate
        aggregated = torch.bmm(adj_norm, node_features)  # (batch, N, in_features)
        
        # Transform
        transformed = self.linear(aggregated)
        transformed = self.activation(transformed)
        
        # Residual + Norm
        output = self.layer_norm(node_features[..., :transformed.size(-1)] + self.dropout(transformed))
        
        return output


class RelationshipClassifier(nn.Module):
    """
    Classify relationships between object pairs
    Predicts both spatial and semantic relationships
    """
    
    def __init__(self, feature_dim=512, num_spatial_rels=8, num_semantic_rels=50):
        super().__init__()
        
        self.num_spatial_rels = num_spatial_rels
        self.num_semantic_rels = num_semantic_rels
        
        # Spatial relationships: above, below, left, right, inside, around, etc.
        self.spatial_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_spatial_rels)
        )
        
        # Semantic relationships: holding, wearing, riding, sitting on, etc.
        self.semantic_classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_semantic_rels)
        )
    
    def forward(self, edge_features):
        """
        Args:
            edge_features: (batch, N, N, feature_dim)
        
        Returns:
            spatial_logits: (batch, N, N, num_spatial_rels)
            semantic_logits: (batch, N, N, num_semantic_rels)
        """
        spatial_logits = self.spatial_classifier(edge_features)
        semantic_logits = self.semantic_classifier(edge_features)
        
        return spatial_logits, semantic_logits


class MessagePassingBlock(nn.Module):
    """
    Message Passing Block combining attention and GCN
    """
    
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadGraphAttention(d_model, num_heads, dropout)
        self.gcn = GraphConvolutionalLayer(d_model, d_model, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, node_features, edge_features, adjacency_matrix=None):
        """
        Args:
            node_features: (batch, N, d_model)
            edge_features: (batch, N, N, d_model)
            adjacency_matrix: (batch, N, N) optional
        
        Returns:
            updated_node_features: (batch, N, d_model)
        """
        # Self-attention with edge features
        attn_output, _ = self.attention(node_features, node_features, node_features, edge_features)
        
        # Graph convolution
        if adjacency_matrix is not None:
            gcn_output = self.gcn(attn_output, adjacency_matrix)
        else:
            # Use uniform adjacency if not provided
            N = node_features.size(1)
            adj = torch.ones(N, N, device=node_features.device)
            gcn_output = self.gcn(attn_output, adj)
        
        # Feed-forward
        ffn_output = self.ffn(gcn_output)
        output = self.layer_norm(gcn_output + ffn_output)
        
        return output


class VisualRelationshipGraph(nn.Module):
    """
    Complete Visual Relationship Graph Module
    
    Pipeline:
    1. Extract spatial features between objects
    2. Encode object features from Faster R-CNN
    3. Message passing to update node representations
    4. Classify relationships (spatial + semantic)
    5. Output enriched object features for caption generation
    """
    
    def __init__(
        self,
        object_feature_dim=512,
        hidden_dim=512,
        num_heads=8,
        num_layers=3,
        num_spatial_rels=8,
        num_semantic_rels=50,
        dropout=0.1
    ):
        super().__init__()
        
        self.object_feature_dim = object_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Spatial feature extractor
        self.spatial_extractor = SpatialFeatureExtractor(
            input_dim=11,
            hidden_dim=256,
            output_dim=hidden_dim
        )
        
        # Object feature encoder
        self.object_encoder = nn.Sequential(
            nn.Linear(object_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Edge feature fusion (combine spatial + visual features)
        self.edge_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Message passing layers
        self.message_passing_layers = nn.ModuleList([
            MessagePassingBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Relationship classifier
        self.relationship_classifier = RelationshipClassifier(
            hidden_dim, num_spatial_rels, num_semantic_rels
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def compute_visual_edge_features(self, node_features):
        """
        Compute visual similarity between object pairs
        
        Args:
            node_features: (batch, N, hidden_dim)
        
        Returns:
            visual_edges: (batch, N, N, hidden_dim)
        """
        batch_size, N, dim = node_features.shape
        
        # Compute pairwise concatenation
        node_i = node_features.unsqueeze(2).expand(batch_size, N, N, dim)  # (B, N, N, dim)
        node_j = node_features.unsqueeze(1).expand(batch_size, N, N, dim)  # (B, N, N, dim)
        
        # Concatenate
        visual_edges = torch.cat([node_i, node_j], dim=-1)  # (B, N, N, 2*dim)
        
        return visual_edges
    
    def forward(self, object_features, boxes, return_relationships=False):
        """
        Forward pass through the relationship graph
        
        Args:
            object_features: (batch, N, object_feature_dim) - features from Faster R-CNN
            boxes: (batch, N, 4) - bounding boxes [x1, y1, x2, y2]
            return_relationships: bool - whether to return relationship predictions
        
        Returns:
            enriched_features: (batch, N, hidden_dim) - object features enhanced with relationships
            relationship_preds: dict (optional) - spatial and semantic relationship predictions
        """
        batch_size, N, _ = object_features.shape
        
        # 1. Encode object features
        node_features = self.object_encoder(object_features)  # (batch, N, hidden_dim)
        
        # 2. Extract spatial features (process each batch item)
        spatial_features_list = []
        for i in range(batch_size):
            spatial_feat = self.spatial_extractor(boxes[i])  # (N, N, hidden_dim)
            spatial_features_list.append(spatial_feat)
        spatial_features = torch.stack(spatial_features_list, dim=0)  # (batch, N, N, hidden_dim)
        
        # 3. Compute visual edge features
        visual_edges = self.compute_visual_edge_features(node_features)  # (batch, N, N, 2*hidden_dim)
        
        # 4. Fuse spatial and visual features
        edge_features = self.edge_fusion(visual_edges)  # (batch, N, N, hidden_dim)
        edge_features = edge_features + spatial_features  # Residual connection
        
        # 5. Message passing
        for layer in self.message_passing_layers:
            node_features = layer(node_features, edge_features)
        
        # 6. Classify relationships (if needed)
        relationship_preds = None
        if return_relationships:
            spatial_logits, semantic_logits = self.relationship_classifier(edge_features)
            relationship_preds = {
                'spatial': spatial_logits,    # (batch, N, N, num_spatial_rels)
                'semantic': semantic_logits   # (batch, N, N, num_semantic_rels)
            }
        
        # 7. Output projection
        enriched_features = self.output_proj(node_features)
        
        return enriched_features, relationship_preds
    
    def get_relationship_labels(self):
        """Return human-readable relationship labels"""
        spatial_rels = [
            'above', 'below', 'left', 'right',
            'inside', 'around', 'overlapping', 'adjacent'
        ]
        
        semantic_rels = [
            'holding', 'wearing', 'riding', 'sitting_on',
            'standing_on', 'lying_on', 'walking_on', 'running_on',
            'looking_at', 'touching', 'carrying', 'using',
            'eating', 'drinking', 'playing', 'reading',
            'watching', 'listening', 'talking_to', 'next_to',
            'in_front_of', 'behind', 'near', 'far_from',
            'part_of', 'attached_to', 'hanging_from', 'covering',
            'surrounded_by', 'contains', 'supported_by', 'leaning_on',
            'mounted_on', 'painted_on', 'printed_on', 'reflected_in',
            'similar_to', 'different_from', 'belongs_to', 'owned_by',
            'made_of', 'same_as', 'interacting_with', 'connected_to',
            'separated_from', 'grouped_with', 'aligned_with', 'facing',
            'parallel_to', 'perpendicular_to'
        ]
        
        return spatial_rels, semantic_rels


class RelationshipGraphLoss(nn.Module):
    """
    Loss function for training the relationship graph
    Combines spatial and semantic relationship losses
    """
    
    def __init__(self, spatial_weight=1.0, semantic_weight=1.0):
        super().__init__()
        self.spatial_weight = spatial_weight
        self.semantic_weight = semantic_weight
        
        self.spatial_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with 'spatial' and 'semantic' logits
            targets: dict with 'spatial' and 'semantic' labels
        
        Returns:
            total_loss: scalar tensor
            loss_dict: dict of individual losses
        """
        spatial_logits = predictions['spatial']  # (batch, N, N, num_spatial_rels)
        semantic_logits = predictions['semantic']  # (batch, N, N, num_semantic_rels)
        
        spatial_targets = targets['spatial']  # (batch, N, N)
        semantic_targets = targets['semantic']  # (batch, N, N)
        
        # Flatten for cross-entropy
        batch_size, N, _, num_spatial = spatial_logits.shape
        _, _, _, num_semantic = semantic_logits.shape
        
        spatial_logits_flat = spatial_logits.view(-1, num_spatial)
        semantic_logits_flat = semantic_logits.view(-1, num_semantic)
        
        spatial_targets_flat = spatial_targets.view(-1)
        semantic_targets_flat = semantic_targets.view(-1)
        
        # Compute losses
        spatial_loss = self.spatial_criterion(spatial_logits_flat, spatial_targets_flat)
        semantic_loss = self.semantic_criterion(semantic_logits_flat, semantic_targets_flat)
        
        # Total loss
        total_loss = self.spatial_weight * spatial_loss + self.semantic_weight * semantic_loss
        
        loss_dict = {
            'spatial_loss': spatial_loss.item(),
            'semantic_loss': semantic_loss.item(),
            'total_relationship_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


# ============================================================================
# Utility Functions
# ============================================================================

def build_adjacency_matrix(boxes, threshold=0.5, k_neighbors=5):
    """
    Build adjacency matrix based on spatial proximity
    
    Args:
        boxes: (N, 4) bounding boxes
        threshold: IoU threshold for connection
        k_neighbors: connect to k nearest neighbors
    
    Returns:
        adjacency: (N, N) binary matrix
    """
    N = boxes.size(0)
    
    # Compute centers
    centers = torch.stack([
        (boxes[:, 0] + boxes[:, 2]) / 2,
        (boxes[:, 1] + boxes[:, 3]) / 2
    ], dim=1)  # (N, 2)
    
    # Compute pairwise distances
    dist_matrix = torch.cdist(centers, centers)  # (N, N)
    
    # Connect k nearest neighbors
    _, indices = torch.topk(dist_matrix, k=min(k_neighbors + 1, N), dim=1, largest=False)
    
    adjacency = torch.zeros(N, N, device=boxes.device)
    for i in range(N):
        adjacency[i, indices[i]] = 1
    
    # Make symmetric
    adjacency = (adjacency + adjacency.t()) > 0
    adjacency = adjacency.float()
    
    return adjacency


def visualize_relationship_graph(boxes, relationships, labels, save_path=None):
    """
    Visualize the relationship graph
    
    Args:
        boxes: (N, 4) bounding boxes
        relationships: (N, N) relationship matrix
        labels: list of N object labels
        save_path: path to save visualization
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("Matplotlib and NetworkX required for visualization")
        return
    
    N = boxes.size(0)
    
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(N):
        G.add_node(i, label=labels[i])
    
    # Add edges
    for i in range(N):
        for j in range(N):
            if i != j and relationships[i, j] > 0:
                G.add_edge(i, j, weight=relationships[i, j].item())
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, arrows=True)
    nx.draw_networkx_labels(G, pos, {i: labels[i] for i in range(N)}, font_size=10)
    
    plt.title("Visual Relationship Graph")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Testing and Debugging
# ============================================================================

def test_relationship_graph():
    """Test the relationship graph module"""
    print("Testing Visual Relationship Graph...")
    
    batch_size = 2
    num_objects = 5
    object_feature_dim = 512
    
    # Create dummy data
    object_features = torch.randn(batch_size, num_objects, object_feature_dim)
    boxes = torch.rand(batch_size, num_objects, 4) * 224
    boxes[:, :, 2:] += boxes[:, :, :2]  # Ensure x2 > x1, y2 > y1
    
    # Create model
    model = VisualRelationshipGraph(
        object_feature_dim=512,
        hidden_dim=512,
        num_heads=8,
        num_layers=3,
        num_spatial_rels=8,
        num_semantic_rels=50
    )
    
    # Forward pass
    enriched_features, relationships = model(
        object_features, boxes, return_relationships=True
    )
    
    print(f"Input features: {object_features.shape}")
    print(f"Enriched features: {enriched_features.shape}")
    print(f"Spatial relationships: {relationships['spatial'].shape}")
    print(f"Semantic relationships: {relationships['semantic'].shape}")
    
    # Test loss
    criterion = RelationshipGraphLoss()
    
    # Dummy targets
    targets = {
        'spatial': torch.randint(0, 8, (batch_size, num_objects, num_objects)),
        'semantic': torch.randint(0, 50, (batch_size, num_objects, num_objects))
    }
    
    loss, loss_dict = criterion(relationships, targets)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")
    
    print("\n✓ All tests passed!")


if __name__ == '__main__':
    test_relationship_graph()
