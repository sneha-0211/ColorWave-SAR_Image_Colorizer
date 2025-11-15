"""
Multi-Branch Generator with Attention and Wavelet Features
Advanced generator architecture for SAR image colorization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional wavelet support - if pywt is not available, wavelet features will be disabled
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    import warnings
    warnings.warn("PyWavelets (pywt) not available. Wavelet features will be disabled. Install with: pip install PyWavelets")


class WaveletTransform(nn.Module):
    """Wavelet transform layer for multi-scale feature extraction"""
    
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet
        self.use_wavelet = PYWT_AVAILABLE
        
    def forward(self, x):
        if not self.use_wavelet:
            # If pywt is not available, return downsampled version as a fallback
            batch_size, channels, height, width = x.shape
            # Simple downsampling as fallback
            ll = F.avg_pool2d(x, kernel_size=2)
            return ll, ll, ll, ll
        
        # Apply wavelet transform to each channel
        coeffs_list = []
        for i in range(x.shape[1]):
            # Detach to break autograd graph before NumPy conversion
            coeffs = pywt.dwt2(x[:, i].detach().cpu().numpy(), self.wavelet, mode='symmetric')
            coeffs_list.append(coeffs)
        
        # Reconstruct in PyTorch
        batch_size, channels, height, width = x.shape
        new_height, new_width = height // 2, width // 2
        
        # Stack coefficients
        ll = torch.stack([torch.from_numpy(coeffs[0]) for coeffs in coeffs_list], dim=1).to(x.device)
        lh = torch.stack([torch.from_numpy(coeffs[1][0]) for coeffs in coeffs_list], dim=1).to(x.device)
        hl = torch.stack([torch.from_numpy(coeffs[1][1]) for coeffs in coeffs_list], dim=1).to(x.device)
        hh = torch.stack([torch.from_numpy(coeffs[1][2]) for coeffs in coeffs_list], dim=1).to(x.device)
        
        return ll, lh, hl, hh


class WaveletInverseTransform(nn.Module):
    """Inverse wavelet transform layer"""
    
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet
        self.use_wavelet = PYWT_AVAILABLE
        
    def forward(self, ll, lh, hl, hh):
        if not self.use_wavelet:
            # If pywt is not available, use upsampling as fallback
            return F.interpolate(ll, scale_factor=2, mode='bilinear', align_corners=False)
        
        batch_size, channels = ll.shape[:2]
        reconstructed = []
        
        for i in range(channels):
            coeffs = (
                ll[:, i].detach().cpu().numpy(),
                (
                    lh[:, i].detach().cpu().numpy(),
                    hl[:, i].detach().cpu().numpy(),
                    hh[:, i].detach().cpu().numpy(),
                ),
            )
            recon = pywt.idwt2(coeffs, self.wavelet, mode='symmetric')
            reconstructed.append(torch.from_numpy(recon).to(ll.device))
        
        return torch.stack(reconstructed, dim=1)


class ResidualBlock(nn.Module):
    """Residual block with spectral normalization"""
    
    def __init__(self, in_channels, out_channels, use_sn=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if use_sn:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)
        
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        if use_sn and in_channels != out_channels:
            self.skip = nn.utils.spectral_norm(self.skip)
        
    def forward(self, x):
        residual = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Avoid in-place addition to keep autograd graph intact
        out = out + residual
        return F.relu(out)


class SelfAttention(nn.Module):
    """Self-attention mechanism for long-range dependencies"""
    
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Generate query, key, value
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        
        # Compute attention
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x


class MultiBranchGenerator(nn.Module):
    """
    Multi-branch generator with attention and wavelet features
    """
    
    def __init__(self, in_channels=1, out_channels=3, base_channels=64, 
                 num_branches=3, use_attention=True, use_wavelet=True):
        super().__init__()
        
        self.num_branches = num_branches
        self.use_attention = use_attention
        self.use_wavelet = use_wavelet
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=False)
        )
        
        # Multi-branch encoder
        self.branches = nn.ModuleList()
        for i in range(num_branches):
            branch = nn.ModuleList([
                ResidualBlock(base_channels, base_channels * 2),
                ResidualBlock(base_channels * 2, base_channels * 4),
                ResidualBlock(base_channels * 4, base_channels * 8),
            ])
            self.branches.append(branch)
        
        # Attention modules
        if use_attention:
            self.attention_modules = nn.ModuleList([
                SelfAttention(base_channels * 8) for _ in range(num_branches)
            ])
        
        # Wavelet processing
        if use_wavelet:
            self.wavelet_transform = WaveletTransform()
            self.wavelet_inverse = WaveletInverseTransform()
            self.wavelet_conv = nn.Conv2d(base_channels * 8 * 4, base_channels * 8, 1)
        
        # Fusion module
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels * 8 * num_branches, base_channels * 8, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling layers
        self.upsample = nn.ModuleList([
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2),
        ])
        
        # Decoder
        self.decoder = nn.ModuleList([
            ResidualBlock(base_channels * 4, base_channels * 4),
            ResidualBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels, base_channels),
        ])
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Initial processing
        x = self.initial_conv(x)
        
        # Multi-branch processing
        branch_outputs = []
        for i, branch in enumerate(self.branches):
            branch_out = x
            for layer in branch:
                branch_out = layer(branch_out)
                branch_out = F.max_pool2d(branch_out, 2)
            
            # Apply attention
            if self.use_attention:
                branch_out = self.attention_modules[i](branch_out)
            
            # Apply wavelet transform
            if self.use_wavelet:
                ll, lh, hl, hh = self.wavelet_transform(branch_out)
                wavelet_features = torch.cat([ll, lh, hl, hh], dim=1)
                wavelet_features = self.wavelet_conv(wavelet_features)
                # Bring wavelet feature maps back to the current feature spatial size
                if wavelet_features.shape[2:] != branch_out.shape[2:]:
                    wavelet_features = F.interpolate(
                        wavelet_features,
                        size=branch_out.shape[2:],
                        mode='bilinear',
                        align_corners=False,
                    )
                branch_out = branch_out + wavelet_features
            
            branch_outputs.append(branch_out)
        
        # Fusion
        fused = torch.cat(branch_outputs, dim=1)
        fused = self.fusion(fused)
        
        # Decoder
        for i, (upsample, decoder) in enumerate(zip(self.upsample, self.decoder)):
            fused = upsample(fused)
            fused = decoder(fused)
        
        # Final output
        output = self.final_conv(fused)
        
        return output
    
    def get_model_size(self):
        """Get model size in parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GeneratorLight(nn.Module):
    """
    Lightweight generator for faster inference
    """
    
    def __init__(self, in_channels=1, out_channels=3, base_channels=32):
        super().__init__()
        
        # Encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 4, 2, 1),
                nn.BatchNorm2d(base_channels),
                nn.LeakyReLU(0.2, inplace=False)
            ),
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
                nn.BatchNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2, inplace=False)
            ),
            nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
                nn.BatchNorm2d(base_channels * 4),
                nn.LeakyReLU(0.2, inplace=False)
            ),
            nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
                nn.BatchNorm2d(base_channels * 8),
                nn.LeakyReLU(0.2, inplace=False)
            )
        ])
        
        # Decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=False)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(inplace=False)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=False)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(base_channels, out_channels, 4, 2, 1),
                nn.Tanh()
            )
        ])
        
    def forward(self, x):
        # Encoder
        for layer in self.encoder:
            x = layer(x)
        
        # Decoder
        for layer in self.decoder:
            x = layer(x)
        
        return x
