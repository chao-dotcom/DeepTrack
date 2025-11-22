"""Person Re-Identification Network"""
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F


class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class ReIDModel(nn.Module):
    """
    Person Re-Identification Network
    Based on ResNet50 with attention mechanisms
    """
    def __init__(self, num_classes: int = 751, feature_dim: int = 2048, dropout: float = 0.5):
        super().__init__()
        
        # Backbone: ResNet50
        try:
            resnet = models.resnet50(pretrained=True)
        except Exception:
            # Fallback if pretrained weights unavailable
            resnet = models.resnet50(pretrained=False)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC and avgpool
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Attention Module
        self.attention = ChannelAttention(2048)
        
        # Feature embedding
        self.bn = nn.BatchNorm1d(feature_dim)
        self.bn.bias.requires_grad_(False)
        
        # Classifier (for training with classification loss)
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self._init_params()
    
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features: bool = False):
        """
        Args:
            x: Input tensor (B, 3, 256, 128)
            return_features: If True, return features instead of logits
        
        Returns:
            If return_features: (B, feature_dim) normalized features
            Else: (B, num_classes) classification logits, (B, feature_dim) features
        """
        # Extract features
        x = self.backbone(x)  # (B, 2048, 8, 4)
        
        # Apply attention
        x = self.attention(x)
        
        # Global pooling
        x = self.gap(x)  # (B, 2048, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 2048)
        
        # Normalize features
        features = self.bn(x)
        
        if return_features:
            # L2 normalize for cosine similarity
            return F.normalize(features, p=2, dim=1)
        
        # Classification
        features = self.dropout(features)
        logits = self.classifier(features)
        
        return logits, features


class TripletLoss(nn.Module):
    """Triplet loss with hard mining"""
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor, positive, negative: (B, feature_dim) L2-normalized features
        """
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class CombinedLoss(nn.Module):
    """Combined triplet + classification loss"""
    def __init__(self, num_classes: int, margin: float = 0.3, 
                 lambda_triplet: float = 1.0, lambda_cls: float = 1.0):
        super().__init__()
        self.triplet_loss = TripletLoss(margin)
        try:
            # Try label smoothing if available
            self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        except TypeError:
            # Fallback for older PyTorch versions
            self.classification_loss = nn.CrossEntropyLoss()
        self.lambda_triplet = lambda_triplet
        self.lambda_cls = lambda_cls
    
    def forward(self, logits, features, labels, anchor, positive, negative):
        """
        Args:
            logits: (B, num_classes) classification logits
            features: (B, feature_dim) features before normalization
            labels: (B,) person IDs
            anchor, positive, negative: (B, feature_dim) normalized features
        """
        triplet_loss = self.triplet_loss(anchor, positive, negative)
        cls_loss = self.classification_loss(logits, labels)
        
        total_loss = (self.lambda_triplet * triplet_loss + 
                     self.lambda_cls * cls_loss)
        
        return total_loss, triplet_loss, cls_loss


