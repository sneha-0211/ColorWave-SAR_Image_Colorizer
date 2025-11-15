"""
Comprehensive Loss Functions for SAR Image Colorization
Production-ready loss implementations with proper error handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from skimage.metrics import structural_similarity as ssim
import numpy as np


class L1Loss(nn.Module):
    """L1 Loss with optional masking"""
    
    def __init__(self, reduction='mean', mask_threshold=None):
        super().__init__()
        self.reduction = reduction
        self.mask_threshold = mask_threshold
        
    def forward(self, pred, target):
        if self.mask_threshold is not None:
            mask = target > self.mask_threshold
            loss = F.l1_loss(pred, target, reduction='none')
            loss = loss * mask.float()
            if self.reduction == 'mean':
                return loss.sum() / mask.sum().clamp(min=1)
            elif self.reduction == 'sum':
                return loss.sum()
        else:
            return F.l1_loss(pred, target, reduction=self.reduction)


class L2Loss(nn.Module):
    """L2 Loss (MSE) with optional masking"""
    
    def __init__(self, reduction='mean', mask_threshold=None):
        super().__init__()
        self.reduction = reduction
        self.mask_threshold = mask_threshold
        
    def forward(self, pred, target):
        if self.mask_threshold is not None:
            mask = target > self.mask_threshold
            loss = F.mse_loss(pred, target, reduction='none')
            loss = loss * mask.float()
            if self.reduction == 'mean':
                return loss.sum() / mask.sum().clamp(min=1)
            elif self.reduction == 'sum':
                return loss.sum()
        else:
            return F.mse_loss(pred, target, reduction=self.reduction)


class SSIMLoss(nn.Module):
    """SSIM Loss implementation with dynamic channel support"""
    
    def __init__(self, window_size=11, size_average=True, channel=None):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel  # None means auto-detect from input
        
    def forward(self, pred, target):
        return 1 - self._ssim(pred, target)
    
    def _ssim(self, pred, target):
        # Auto-detect number of channels if not specified
        num_channels = self.channel if self.channel is not None else pred.shape[1]
        
        # Ensure we don't exceed the actual number of channels
        num_channels = min(num_channels, pred.shape[1], target.shape[1])
        
        # Compute SSIM for each channel
        ssim_values = []
        for i in range(num_channels):
            pred_ch = pred[:, i:i+1, :, :]
            target_ch = target[:, i:i+1, :, :]
            ssim_val = self._compute_ssim(pred_ch, target_ch)
            ssim_values.append(ssim_val)
        
        if len(ssim_values) > 1:
            ssim_tensor = torch.stack(ssim_values, dim=0)
        else:
            ssim_tensor = ssim_values[0]
        
        if self.size_average:
            return ssim_tensor.mean()
        else:
            return ssim_tensor
    
    def _compute_ssim(self, pred, target):
        mu1 = F.avg_pool2d(pred, self.window_size, 1, self.window_size//2)
        mu2 = F.avg_pool2d(target, self.window_size, 1, self.window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(pred * pred, self.window_size, 1, self.window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, self.window_size, 1, self.window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, self.window_size, 1, self.window_size//2) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    
    def __init__(self, device='cpu', feature_layers=[3, 8, 15, 22]):
        super().__init__()
        self.device = device
        self.feature_layers = feature_layers
        
        # Load VGG16 features with offline-safe fallback
        # Try to use pretrained weights; if unavailable (no internet or missing weights),
        # fall back to randomly initialized weights to keep training runnable.
        try:
            try:
                from torchvision.models import VGG16_Weights
                vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
            except ImportError:
                # Older torchvision API
                vgg = models.vgg16(pretrained=True).features
        except Exception as e:
            print(
                f"[PerceptualLoss] Warning: Could not load pretrained VGG16 weights (reason: {e}). "
                "Falling back to untrained VGG16 features. Perceptual loss effectiveness may be reduced."
            )
            try:
                # Newer API without weights (no download)
                vgg = models.vgg16(weights=None).features
            except TypeError:
                # Older API uses pretrained flag
                vgg = models.vgg16(pretrained=False).features

        for layer in vgg.modules():
            if isinstance(layer, nn.ReLU):
                layer.inplace = False

        self.vgg_layers = nn.ModuleList()
        
        for i, layer in enumerate(vgg):
            if i in feature_layers:
                self.vgg_layers.append(layer)
            if i > max(feature_layers):
                break
        
        # Freeze VGG parameters
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        
        self.vgg_layers = self.vgg_layers.to(device)
        self.criterion = nn.L1Loss()
        
    def forward(self, pred, target):
        # Ensure inputs are in [0, 1] range
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # Convert to VGG input format (3 channels, normalized)
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
        
        # Normalize to ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        pred = (pred - mean) / std
        target = (target - mean) / std
        
        # Extract features
        pred_features = []
        target_features = []
        
        for layer in self.vgg_layers:
            pred = layer(pred)
            target = layer(target)
            pred_features.append(pred)
            target_features.append(target)
        
        # Compute perceptual loss
        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += self.criterion(pred_feat, target_feat)
        
        return loss / len(pred_features)


class GANLoss(nn.Module):
    """GAN loss with multiple loss types"""
    
    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.gan_mode = gan_mode
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.loss = None  # Wasserstein loss is computed differently
        else:
            raise ValueError(f"Unsupported GAN mode: {gan_mode}")
    
    def forward(self, prediction, target_is_real):
        if self.gan_mode == 'wgangp':
            if target_is_real:
                return -prediction.mean()
            else:
                return prediction.mean()
        else:
            if target_is_real:
                target_tensor = torch.ones_like(prediction) * self.target_real_label
            else:
                target_tensor = torch.zeros_like(prediction) * self.target_fake_label
            
            return self.loss(prediction, target_tensor)


class GradientPenaltyLoss(nn.Module):
    """Gradient penalty for Wasserstein GAN"""
    
    def __init__(self, lambda_gp=10.0):
        super().__init__()
        self.lambda_gp = lambda_gp
    
    def forward(self, discriminator, real_samples, fake_samples):
        batch_size = real_samples.size(0)
        device = real_samples.device
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, 1).to(device)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)
        
        # Discriminator output
        d_interpolated = discriminator(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return self.lambda_gp * gradient_penalty


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class L1_SSIM_Loss(nn.Module):
    """Combined L1 and SSIM loss"""
    
    def __init__(self, l1_weight=1.0, ssim_weight=1.0, ssim_window_size=11):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss(window_size=ssim_window_size)
    
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        return self.l1_weight * l1 + self.ssim_weight * ssim


class MultiScaleLoss(nn.Module):
    """Multi-scale loss for different resolutions"""
    
    def __init__(self, scales=[1, 0.5, 0.25], loss_type='l1'):
        super().__init__()
        self.scales = scales
        self.loss_type = loss_type
        
        if loss_type == 'l1':
            self.base_loss = L1Loss()
        elif loss_type == 'l2':
            self.base_loss = L2Loss()
        elif loss_type == 'ssim':
            self.base_loss = SSIMLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, pred, target):
        total_loss = 0
        
        for scale in self.scales:
            if scale != 1:
                pred_scaled = F.interpolate(pred, scale_factor=scale, mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
            else:
                pred_scaled = pred
                target_scaled = target
            
            loss = self.base_loss(pred_scaled, target_scaled)
            total_loss += loss
        
        return total_loss / len(self.scales)


class AdversarialLoss(nn.Module):
    """Adversarial loss for GAN training"""
    
    def __init__(self, gan_mode='lsgan', lambda_adv=1.0):
        super().__init__()
        self.gan_loss = GANLoss(gan_mode)
        self.lambda_adv = lambda_adv
    
    def forward(self, pred, target_is_real):
        return self.lambda_adv * self.gan_loss(pred, target_is_real)


class TotalVariationLoss(nn.Module):
    """Total variation loss for smoothness regularization"""
    
    def __init__(self, lambda_tv=1.0):
        super().__init__()
        self.lambda_tv = lambda_tv
    
    def forward(self, x):
        batch_size = x.size(0)
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        return self.lambda_tv * (h_tv + w_tv) / batch_size


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (robust L1)"""
    
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        diff = pred - target
        return torch.sqrt(diff * diff + self.epsilon * self.epsilon).mean()


class EdgeLoss(nn.Module):
    """Edge loss for preserving image structures"""
    
    def __init__(self, lambda_edge=1.0, epsilon=1e-6):
        super().__init__()
        self.lambda_edge = lambda_edge
        self.epsilon = epsilon
        
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred, target):
        # Handle multi-channel images by computing edges for each channel
        if pred.shape[1] > 1:
            # For multi-channel images, compute edges for each channel separately
            pred_edges = []
            target_edges = []
            
            for c in range(pred.shape[1]):
                pred_ch = pred[:, c:c+1, :, :]
                target_ch = target[:, c:c+1, :, :]
                
                pred_edges_x = F.conv2d(pred_ch, self.sobel_x, padding=1)
                pred_edges_y = F.conv2d(pred_ch, self.sobel_y, padding=1)
                pred_edges.append(torch.sqrt(pred_edges_x**2 + pred_edges_y**2 + self.epsilon))
                
                target_edges_x = F.conv2d(target_ch, self.sobel_x, padding=1)
                target_edges_y = F.conv2d(target_ch, self.sobel_y, padding=1)
                target_edges.append(torch.sqrt(target_edges_x**2 + target_edges_y**2 + self.epsilon))
            
            pred_edges = torch.cat(pred_edges, dim=1)
            target_edges = torch.cat(target_edges, dim=1)
        else:
            # Single channel case
            pred_edges_x = F.conv2d(pred, self.sobel_x, padding=1)
            pred_edges_y = F.conv2d(pred, self.sobel_y, padding=1)
            pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2 + self.epsilon)
            
            target_edges_x = F.conv2d(target, self.sobel_x, padding=1)
            target_edges_y = F.conv2d(target, self.sobel_y, padding=1)
            target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2 + self.epsilon)
        
        return self.lambda_edge * F.l1_loss(pred_edges, target_edges)


class CombinedLoss(nn.Module):
    """Combined loss function with multiple components"""
    
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = {
                'l1': 1.0,
                'ssim': 1.0,
                'perceptual': 0.1,
                'adversarial': 0.1,
                'tv': 0.01,
                'edge': 0.1
            }
        
        self.weights = weights
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = PerceptualLoss()
        self.gan_loss = GANLoss()
        self.tv_loss = TotalVariationLoss()
        self.edge_loss = EdgeLoss()
    
    def forward(self, pred, target, discriminator_output=None, target_is_real=None):
        total_loss = 0
        loss_dict = {}
        
        # L1 loss
        l1 = self.l1_loss(pred, target)
        total_loss += self.weights['l1'] * l1
        loss_dict['l1'] = l1.item()
        
        # SSIM loss
        ssim = self.ssim_loss(pred, target)
        total_loss += self.weights['ssim'] * ssim
        loss_dict['ssim'] = ssim.item()
        
        # Perceptual loss
        perceptual = self.perceptual_loss(pred, target)
        total_loss += self.weights['perceptual'] * perceptual
        loss_dict['perceptual'] = perceptual.item()
        
        # Adversarial loss
        if discriminator_output is not None and target_is_real is not None:
            adv = self.gan_loss(discriminator_output, target_is_real)
            total_loss += self.weights['adversarial'] * adv
            loss_dict['adversarial'] = adv.item()
        
        # Total variation loss
        tv = self.tv_loss(pred)
        total_loss += self.weights['tv'] * tv
        loss_dict['tv'] = tv.item()
        
        # Edge loss
        edge = self.edge_loss(pred, target)
        total_loss += self.weights['edge'] * edge
        loss_dict['edge'] = edge.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict