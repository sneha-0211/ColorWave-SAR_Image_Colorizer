"""
Advanced Discriminator Architectures for SAR Image Colorization
Multiple discriminator types for different training strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for local feature discrimination
    """
    
    def __init__(self, in_channels=4, base_channels=64, num_layers=3, use_sn=True):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_sn = use_sn
        
        # Build discriminator layers
        layers = []
        in_ch = in_channels
        
        for i in range(num_layers):
            out_ch = base_channels * (2 ** i)
            layers.append(self._make_layer(in_ch, out_ch, use_sn))
            in_ch = out_ch
        
        # Final layer
        layers.append(nn.Conv2d(in_ch, 1, 4, 1, 0))
        if use_sn:
            layers[-1] = nn.utils.spectral_norm(layers[-1])
        
        self.discriminator = nn.Sequential(*layers)
        
    def _make_layer(self, in_channels, out_channels, use_sn):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=False)
        )
        
        if use_sn:
            layer[0] = nn.utils.spectral_norm(layer[0])
        
        return layer
    
    def forward(self, x):
        return self.discriminator(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for different resolution features
    """
    
    def __init__(self, in_channels=4, base_channels=64, num_scales=3, use_sn=True):
        super().__init__()
        
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()
        
        for i in range(num_scales):
            scale_factor = 2 ** i
            disc = PatchDiscriminator(
                in_channels=in_channels,
                base_channels=base_channels,
                num_layers=3,
                use_sn=use_sn
            )
            self.discriminators.append(disc)
    
    def forward(self, x):
        outputs = []
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = F.avg_pool2d(x, 2, 2)
            outputs.append(disc(x))
        return outputs


class GlobalDiscriminator(nn.Module):
    """
    Global discriminator for full image discrimination
    """
    
    def __init__(self, in_channels=4, base_channels=64, use_sn=True):
        super().__init__()
        
        self.use_sn = use_sn
        
        # Feature extraction
        self.features = nn.Sequential(
            self._conv_block(in_channels, base_channels, use_sn),
            self._conv_block(base_channels, base_channels * 2, use_sn),
            self._conv_block(base_channels * 2, base_channels * 4, use_sn),
            self._conv_block(base_channels * 4, base_channels * 8, use_sn),
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(base_channels * 4, 1)
        )
        
    def _conv_block(self, in_channels, out_channels, use_sn):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=False)
        )
        
        if use_sn:
            block[0] = nn.utils.spectral_norm(block[0])
        
        return block
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SpectralDiscriminator(nn.Module):
    """
    Discriminator with spectral normalization and attention
    """
    
    def __init__(self, in_channels=4, base_channels=64, use_attention=True):
        super().__init__()
        
        self.use_attention = use_attention
        
        # Initial layers
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Main feature extraction
        self.features = nn.ModuleList([
            self._make_block(base_channels, base_channels * 2),
            self._make_block(base_channels * 2, base_channels * 4),
            self._make_block(base_channels * 4, base_channels * 8),
            self._make_block(base_channels * 8, base_channels * 8),
        ])
        
        # Attention module
        if use_attention:
            self.attention = self._make_attention(base_channels * 8)
        
        # Final classification
        self.final = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0)
        )
        
    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=False)
        )
    
    def _make_attention(self, channels):
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(channels, channels // 8, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(channels // 8, channels, 1)),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.initial(x)
        
        for layer in self.features:
            x = layer(x)
        
        if self.use_attention:
            attention = self.attention(x)
            x = x * attention
        
        x = self.final(x)
        return x


class WassersteinDiscriminator(nn.Module):
    """
    Wasserstein GAN discriminator (critic)
    """
    
    def __init__(self, in_channels=4, base_channels=64, num_layers=4):
        super().__init__()
        
        layers = []
        in_ch = in_channels
        
        for i in range(num_layers):
            out_ch = base_channels * (2 ** i)
            layers.append(nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=False)
            ))
            in_ch = out_ch
        
        # Final layer
        layers.append(nn.utils.spectral_norm(nn.Conv2d(in_ch, 1, 4, 1, 0)))
        
        self.discriminator = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.discriminator(x)


class EnsembleDiscriminator(nn.Module):
    """
    Ensemble of multiple discriminators
    """
    
    def __init__(self, in_channels=4, base_channels=64, num_discriminators=3):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_channels, base_channels, use_sn=True)
            for _ in range(num_discriminators)
        ])
        
    def forward(self, x):
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(x))
        return outputs
    
    def get_loss(self, real_outputs, fake_outputs):
        """Compute ensemble loss"""
        total_loss = 0
        for real_out, fake_out in zip(real_outputs, fake_outputs):
            real_loss = F.binary_cross_entropy_with_logits(real_out, torch.ones_like(real_out))
            fake_loss = F.binary_cross_entropy_with_logits(fake_out, torch.zeros_like(fake_out))
            total_loss += real_loss + fake_loss
        return total_loss / len(self.discriminators)


class AdaptiveDiscriminator(nn.Module):
    """
    Adaptive discriminator that adjusts based on training progress
    """
    
    def __init__(self, in_channels=4, base_channels=64, max_layers=5):
        super().__init__()
        
        self.max_layers = max_layers
        self.current_layers = 1
        
        # Progressive layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_channels, base_channels, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=False)
            )
        ])
        
        for i in range(1, max_layers):
            in_ch = base_channels * (2 ** (i - 1))
            out_ch = base_channels * (2 ** i)
            self.layers.append(nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 4, 2, 1)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        
        # Final layer
        self.final = nn.utils.spectral_norm(
            nn.Conv2d(base_channels * (2 ** (max_layers - 1)), 1, 4, 1, 0)
        )
    
    def add_layer(self):
        """Add a new layer during progressive training"""
        if self.current_layers < self.max_layers:
            self.current_layers += 1
    
    def forward(self, x):
        for i in range(self.current_layers):
            x = self.layers[i](x)
        
        x = self.final(x)
        return x