"""
Classification Models for SAR Image Analysis
MSTAR target recognition and SEN12-FLOOD flood detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional


class SARClassifier(nn.Module):
    """Base SAR image classifier with transfer learning support"""
    
    def __init__(self, num_classes: int, backbone: str = 'resnet18', pretrained: bool = True,
                dropout_rate: float = 0.5, input_channels: int = 1):
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.input_channels = input_channels
        
        # Load backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, 
                                            stride=2, padding=3, bias=False)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, 
                                            stride=2, padding=3, bias=False)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif backbone == 'efficientnet':
            try:
                import torchvision.models as models
                self.backbone = models.efficientnet_b0(pretrained=pretrained)
                # Modify first layer for single channel input
                self.backbone.features[0][0] = nn.Conv2d(input_channels, 32, 
                                                       kernel_size=3, stride=2, padding=1, bias=False)
                self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)
            except ImportError:
                raise ImportError("EfficientNet requires torchvision >= 0.13.0")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        """Forward pass"""
        if self.backbone_name == 'efficientnet':
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.backbone.classifier(x)
        else:
            x = self.backbone(x)
            x = self.dropout(x)
        
        return x


class MSTARClassifier(SARClassifier):
    """MSTAR target recognition classifier"""
    
    def __init__(self, num_classes: int = 10, backbone: str = 'resnet18', 
                 pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            input_channels=1  # MSTAR is single-channel SAR
        )
        
        # MSTAR specific class names
        self.class_names = [
            '2S1', 'BRDM-2', 'BTR-60', 'BTR-70', 'D7', 
            'T62', 'T72', 'ZIL131', 'ZSU234', 'BMP2'
        ]
    
    def get_class_name(self, class_idx: int) -> str:
        """Get class name from index"""
        if 0 <= class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return f"Unknown_{class_idx}"


class FloodDetector(SARClassifier):
    """SEN12-FLOOD flood detection classifier"""
    
    def __init__(self, backbone: str = 'resnet18', pretrained: bool = True, 
                 dropout_rate: float = 0.4):
        super().__init__(
            num_classes=2,  # Binary classification: flood/no-flood
            backbone=backbone,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            input_channels=3  # RGB optical images
        )
        
        # Flood detection class names
        self.class_names = ['No Flood', 'Flood']
    
    def get_class_name(self, class_idx: int) -> str:
        """Get class name from index"""
        if 0 <= class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return f"Unknown_{class_idx}"


class MultiTaskClassifier(nn.Module):
    """Multi-task classifier for both MSTAR and flood detection"""
    
    def __init__(self, mstar_classes: int = 10, backbone: str = 'resnet18', 
                 pretrained: bool = True, dropout_rate: float = 0.4):
        super().__init__()
        
        self.backbone_name = backbone
        
        # Load backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            # Remove final classification layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            feature_dim = self.backbone(dummy_input).view(1, -1).shape[1]
        
        # Task-specific heads
        self.mstar_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, mstar_classes)
        )
        
        self.flood_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)
        )
        
        # Class names
        self.mstar_class_names = [
            '2S1', 'BRDM-2', 'BTR-60', 'BTR-70', 'D7', 
            'T62', 'T72', 'ZIL131', 'ZSU234', 'BMP2'
        ]
        self.flood_class_names = ['No Flood', 'Flood']
    
    def forward(self, x, task: str = 'mstar'):
        """Forward pass for specific task"""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        if task == 'mstar':
            return self.mstar_head(features)
        elif task == 'flood':
            return self.flood_head(features)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def forward_both(self, x):
        """Forward pass for both tasks"""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        mstar_output = self.mstar_head(features)
        flood_output = self.flood_head(features)
        
        return mstar_output, flood_output


class AttentionClassifier(nn.Module):
    """SAR classifier with attention mechanisms"""
    
    def __init__(self, num_classes: int, input_channels: int = 1, 
                 base_channels: int = 64, dropout_rate: float = 0.3):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Encoder with attention
        self.encoder = nn.ModuleList([
            self._make_layer(input_channels, base_channels, 2),
            self._make_layer(base_channels, base_channels * 2, 2),
            self._make_layer(base_channels * 2, base_channels * 4, 2),
            self._make_layer(base_channels * 4, base_channels * 8, 2),
        ])
        
        # Attention modules
        self.attention_modules = nn.ModuleList([
            ChannelAttention(base_channels),
            ChannelAttention(base_channels * 2),
            ChannelAttention(base_channels * 4),
            ChannelAttention(base_channels * 8),
        ])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels * 8, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int):
        """Create a layer with multiple residual blocks"""
        layers = []
        for i in range(num_blocks):
            in_ch = in_channels if i == 0 else out_channels
            layers.append(ResidualBlock(in_ch, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass with attention"""
        for encoder, attention in zip(self.encoder, self.attention_modules):
            x = encoder(x)
            x = attention(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for attention classifier"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """Forward pass"""
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class ChannelAttention(nn.Module):
    """Channel attention module"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass"""
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


def create_classifier(task: str, num_classes: int = None, backbone: str = 'resnet18',
                     pretrained: bool = True, **kwargs) -> nn.Module:
    """Factory function to create appropriate classifier"""
    
    if task.lower() == 'mstar':
        return MSTARClassifier(
            num_classes=num_classes or 10,
            backbone=backbone,
            pretrained=pretrained,
            **kwargs
        )
    elif task.lower() == 'flood':
        return FloodDetector(
            backbone=backbone,
            pretrained=pretrained,
            **kwargs
        )
    elif task.lower() == 'multitask':
        return MultiTaskClassifier(
            mstar_classes=num_classes or 10,
            backbone=backbone,
            pretrained=pretrained,
            **kwargs
        )
    elif task.lower() == 'attention':
        return AttentionClassifier(
            num_classes=num_classes or 10,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def get_model_info(model: nn.Module) -> Dict[str, int]:
    """Get model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }
  
