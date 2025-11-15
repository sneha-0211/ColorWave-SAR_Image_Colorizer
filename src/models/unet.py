"""
Enhanced UNet Architecture for SAR Image Colorization
Production-ready implementation with attention mechanisms and residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout"""
    
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += residual
        return F.relu(out)


class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(attention)


class AttentionBlock(nn.Module):
    """Combined channel and spatial attention"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class UNet(nn.Module):
    """
    Enhanced UNet with attention mechanisms, residual connections, and deep supervision
    """
    
    def __init__(self, in_channels=1, out_channels=3, features=[64, 128, 256, 512, 1024], 
                 dropout_rate=0.1, use_attention=True, use_deep_supervision=True):
        super().__init__()
        
        self.use_attention = use_attention
        self.use_deep_supervision = use_deep_supervision
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        for i, feature in enumerate(features):
            in_ch = in_channels if i == 0 else features[i-1]
            self.encoder.append(ResidualBlock(in_ch, feature, dropout_rate))
            if i < len(features) - 1:
                self.pools.append(nn.MaxPool2d(2, 2))
            if use_attention:
                self.attention_blocks.append(AttentionBlock(feature))
        
        # Bottleneck
        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2, dropout_rate)
        if use_attention:
            self.bottleneck_attention = AttentionBlock(features[-1] * 2)
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.decoder_attention = nn.ModuleList()
        
        for i in range(len(features) - 1, 0, -1):
            # First upconv goes from bottleneck (features[-1] * 2) to features[i-1]
            # Subsequent upconvs go from features[i] to features[i-1]
            if i == len(features) - 1:
                # First decoder level: bottleneck to features[-2]
                self.upconvs.append(nn.ConvTranspose2d(features[i] * 2, features[i-1], 2, 2))
            else:
                # Subsequent decoder levels: features[i] to features[i-1]
                self.upconvs.append(nn.ConvTranspose2d(features[i], features[i-1], 2, 2))
            
            # Decoder input channels: skip connection (features[i-1]) + upconv output (features[i-1])
            decoder_input_channels = features[i-1] + features[i-1]
            self.decoder.append(ResidualBlock(decoder_input_channels, features[i-1], dropout_rate))
            if use_attention:
                self.decoder_attention.append(AttentionBlock(features[i-1]))
        
        # Final layers
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
        self.final_activation = nn.Sigmoid()
        
        # Deep supervision heads
        if use_deep_supervision:
            self.deep_supervision = nn.ModuleList()
            # Deep supervision uses decoder output channels
            # Decoder outputs have features[i-1] channels after processing
            for i in range(len(features) - 1, 0, -1):
                self.deep_supervision.append(nn.Conv2d(features[i-1], out_channels, 1))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        encoder_outputs = []
        for i, (encoder, pool) in enumerate(zip(self.encoder, self.pools + [nn.Identity()])):
            x = encoder(x)
            if self.use_attention and i < len(self.attention_blocks):
                x = self.attention_blocks[i](x)
            encoder_outputs.append(x)
            if i < len(self.pools):
                x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        if self.use_attention:
            x = self.bottleneck_attention(x)
        
        # Decoder
        deep_supervision_outputs = []
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoder)):
            x = upconv(x)
            
            # Skip connection
            skip = encoder_outputs[-(i+2)]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            
            x = decoder(x)
            if self.use_attention:
                x = self.decoder_attention[i](x)
            
            # Deep supervision
            if self.use_deep_supervision:
                deep_out = self.deep_supervision[i](x)
                deep_supervision_outputs.append(deep_out)
        
        # Final output
        x = self.final_conv(x)
        x = self.final_activation(x)
        
        if self.use_deep_supervision and self.training:
            return x, deep_supervision_outputs
        return x
    
    def get_model_size(self):
        """Get model size in parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_memory(self):
        """Estimate model memory usage in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / 1024 / 1024  # MB


class UNetLight(nn.Module):
    """
    Lightweight UNet for faster inference
    """
    
    def __init__(self, in_channels=1, out_channels=3, features=[32, 64, 128, 256]):
        super().__init__()
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        for i, feature in enumerate(features):
            in_ch = in_channels if i == 0 else features[i-1]
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_ch, feature, 3, padding=1),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature, feature, 3, padding=1),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True)
            ))
            if i < len(features) - 1:
                self.pools.append(nn.MaxPool2d(2, 2))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1] * 2, 3, padding=1),
            nn.BatchNorm2d(features[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1] * 2, features[-1] * 2, 3, padding=1),
            nn.BatchNorm2d(features[-1] * 2),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for i in range(len(features) - 1, 0, -1):
            # First upconv needs to come from bottleneck (features[-1] * 2)
            # Subsequent upconvs come from previous decoder output (features[i-1])
            in_channels_up = features[-1] * 2 if i == len(features) - 1 else features[i]
            self.upconvs.append(nn.ConvTranspose2d(in_channels_up, features[i-1], 2, 2))
            # Decoder input channels: skip connection (features[i-1]) + upconv output (features[i-1])
            decoder_input_channels = features[i-1] + features[i-1]
            self.decoder.append(nn.Sequential(
                nn.Conv2d(decoder_input_channels, features[i-1], 3, padding=1),
                nn.BatchNorm2d(features[i-1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(features[i-1], features[i-1], 3, padding=1),
                nn.BatchNorm2d(features[i-1]),
                nn.ReLU(inplace=True)
            ))
        
        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
        self.final_activation = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder
        encoder_outputs = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            encoder_outputs.append(x)
            if i < len(self.pools):
                x = self.pools[i](x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoder)):
            x = upconv(x)
            
            # Skip connection
            skip = encoder_outputs[-(i+2)]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)
        
        # Final output
        x = self.final_conv(x)
        return self.final_activation(x)