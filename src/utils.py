"""
Comprehensive Utility Functions for SAR Image Colorization
Production-ready utilities with error handling and performance optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
from skimage.metrics import structural_similarity as ssim


def seed_everything(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, path: str, **kwargs):
    """Save model checkpoint with comprehensive metadata"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    torch.save(checkpoint, path)


def load_checkpoint(path: str, device: str = 'cpu', model: Optional[nn.Module] = None,
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    """Load model checkpoint with error handling"""
    try:
        checkpoint = torch.load(path, map_location=device)
        
        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
        
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint from {path}: {e}")


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, 
                     metric: str) -> float:
    """Calculate various image quality metrics"""
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    
    if metric.lower() == 'psnr':
        return calculate_psnr(pred, target)
    elif metric.lower() == 'ssim':
        return calculate_ssim(pred, target)
    elif metric.lower() == 'lpips':
        return calculate_lpips(pred, target)
    elif metric.lower() == 'l1':
        return F.l1_loss(pred, target).item()
    elif metric.lower() == 'l2':
        return F.mse_loss(pred, target).item()
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, 
                  window_size: int = 11) -> float:
    """Calculate Structural Similarity Index"""
    # Convert to numpy for SSIM calculation
    pred_np = pred.squeeze(0).numpy().transpose(1, 2, 0)
    target_np = target.squeeze(0).numpy().transpose(1, 2, 0)
    
    # Calculate SSIM for each channel
    ssim_values = []
    for i in range(pred_np.shape[2]):
        ssim_val = ssim(pred_np[:, :, i], target_np[:, :, i], 
                       data_range=1.0, win_size=window_size)
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)


def calculate_lpips(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Learned Perceptual Image Patch Similarity"""
    try:
        import lpips
        lpips_model = lpips.LPIPS(net='alex')
        # Ensure tensors are in [-1, 1] range for LPIPS
        pred_lpips = pred * 2.0 - 1.0
        target_lpips = target * 2.0 - 1.0
        return lpips_model(pred_lpips, target_lpips).item()
    except ImportError:
        # Fallback to L1 distance if LPIPS not available
        return F.l1_loss(pred, target).item()


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array"""
    return tensor.detach().cpu().numpy()


def numpy_to_tensor(array: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """Convert numpy array to tensor"""
    return torch.from_numpy(array).to(device)


def save_image(tensor: torch.Tensor, path: str, normalize: bool = True):
    """Save tensor as image"""
    from PIL import Image
    
    # Convert to numpy
    array = tensor_to_numpy(tensor)
    
    # Handle different tensor shapes
    if len(array.shape) == 4:  # Batch
        array = array[0]
    if len(array.shape) == 3:  # CHW
        array = array.transpose(1, 2, 0)
    
    # Normalize if needed
    if normalize:
        array = (array - array.min()) / (array.max() - array.min())
    
    # Convert to uint8
    array = (array * 255).astype(np.uint8)
    
    # Save
    Image.fromarray(array).save(path)


def load_image(path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load image from file"""
    from PIL import Image
    
    image = Image.open(path)
    
    if target_size:
        image = image.resize(target_size, Image.LANCZOS)
    
    return np.array(image)


def create_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Create logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_device_info() -> Dict[str, Union[str, int, bool]]:
    """Get comprehensive device information"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'device_name': torch.cuda.get_device_name(0),
            'memory_allocated': torch.cuda.memory_allocated(0),
            'memory_reserved': torch.cuda.memory_reserved(0)
        })
    
    return info


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> Dict[str, Union[int, float]]:
    """Get model size information"""
    param_count = count_parameters(model)
    
    # Estimate memory usage
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    return {
        'parameters': param_count,
        'size_bytes': total_size,
        'size_mb': total_size / 1024 / 1024,
        'size_gb': total_size / 1024 / 1024 / 1024
    }


def profile_model(model: nn.Module, input_shape: Tuple[int, ...], 
                 device: str = 'cpu') -> Dict[str, float]:
    """Profile model performance"""
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Profile
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    
    return {
        'avg_inference_time': avg_time,
        'fps': 1.0 / avg_time if avg_time > 0 else 0
    }


def save_config(config: Dict, path: str):
    """Save configuration to file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(path: str) -> Dict:
    """Load configuration from file"""
    with open(path, 'r') as f:
        return json.load(f)


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create experiment directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def setup_experiment_logging(exp_dir: str) -> logging.Logger:
    """Setup logging for experiment"""
    log_file = os.path.join(exp_dir, 'experiment.log')
    return create_logger('experiment', log_file)


def calculate_gradient_norm(model: nn.Module) -> float:
    """Calculate gradient norm for monitoring"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)


def clip_gradients(model: nn.Module, max_norm: float = 1.0):
    """Clip gradients to prevent exploding gradients"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer"""
    return optimizer.param_groups[0]['lr']


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float):
    """Set learning rate for optimizer"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """Create optimizer from configuration"""
    optimizer_type = config['type'].lower()
    lr = config['learning_rate']
    weight_decay = config.get('weight_decay', 0)
    
    if optimizer_type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=config.get('betas', [0.9, 0.999])
        )
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=config.get('betas', [0.9, 0.999])
        )
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=config.get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler from configuration"""
    scheduler_type = config.get('type', '').lower()
    
    if not scheduler_type:
        return None
    
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('T_max', 100),
            eta_min=config.get('eta_min', 0)
        )
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 30),
            gamma=config.get('gamma', 0.1)
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.get('mode', 'min'),
            factor=config.get('factor', 0.5),
            patience=config.get('patience', 10)
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")


def format_time(seconds: float) -> str:
    """Format time in human-readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}m"
    else:
        return f"{seconds/3600:.2f}h"


def format_size(bytes_size: int) -> str:
    """Format size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f}TB"


def validate_config(config: Dict, required_keys: List[str]) -> bool:
    """Validate configuration has required keys"""
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    return True


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """Merge configuration dictionaries"""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged