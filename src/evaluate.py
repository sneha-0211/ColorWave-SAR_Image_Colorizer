"""
Comprehensive Evaluation Script for SAR Image Colorization
Production-ready evaluation with multiple metrics and visualization
Enhanced with classification capabilities for MSTAR/SEN12-FLOOD tasks
"""

import os
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score
import warnings

# Try to import LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    warnings.warn("LPIPS not available. Install with: pip install lpips")

# Support running as a package and as a script
try:
    from .models.unet import UNet, UNetLight
    from .models.generator_adv import MultiBranchGenerator, GeneratorLight
    from .data_pipeline import SARDataset, get_augmentation_pipeline
    from .losses import L1Loss, SSIMLoss, PerceptualLoss, EdgeLoss
    from .utils import seed_everything, load_checkpoint, calculate_metrics
except ImportError:
    from models.unet import UNet, UNetLight
    from models.generator_adv import MultiBranchGenerator, GeneratorLight
    from data_pipeline import SARDataset, get_augmentation_pipeline
    from losses import L1Loss, SSIMLoss, PerceptualLoss, EdgeLoss
    from utils import seed_everything, load_checkpoint, calculate_metrics


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, config: dict, model_path: str, device: str = 'auto'):
        self.config = config
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != 'cpu' else "cpu")
        
        # Setup logging
        self.setup_logging()
        
        # Setup directories
        self.setup_directories()
        
        # Load model
        self.load_model()
        
        # Setup data
        self.setup_data()
        
        # Setup metrics
        self.setup_metrics()
        
    def setup_logging(self):
        """Setup logging"""
        log_dir = Path(self.config['experiment']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Setup output directories"""
        self.output_dir = Path(self.config['experiment']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'metrics').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
    
    def _resolve_model_path(self, model_path: str) -> Optional[str]:
        """Resolve model path, handling both absolute and relative paths"""
        if not model_path:
            return None
        
        # Normalize path separators
        normalized = model_path.replace('\\', os.sep).replace('/', os.sep)
        
        # If path exists as-is, return it
        if os.path.exists(normalized):
            return os.path.abspath(normalized)
        
        # Try as relative path from current working directory
        rel_path = os.path.join(os.getcwd(), normalized.lstrip(os.sep))
        if os.path.exists(rel_path):
            return os.path.abspath(rel_path)
        
        # Try relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # Go up from src/ to project root
        rel_to_project = os.path.join(project_root, normalized.lstrip(os.sep))
        if os.path.exists(rel_to_project):
            return os.path.abspath(rel_to_project)
        
        # If path is like "experiments/checkpoints/best_model.pth", try subdirectories
        if "checkpoints" in normalized and not os.path.exists(normalized):
            # Extract filename
            filename = os.path.basename(normalized)
            # Try supervised and adversarial subdirectories
            for subdir in ["supervised", "adversarial"]:
                # Try relative to project root
                candidate = os.path.join(project_root, "experiments", "checkpoints", subdir, filename)
                if os.path.exists(candidate):
                    return os.path.abspath(candidate)
                # Try relative to current directory
                candidate = os.path.join(os.getcwd(), "experiments", "checkpoints", subdir, filename)
                if os.path.exists(candidate):
                    return os.path.abspath(candidate)
        
        # Try common checkpoint locations
        filename = os.path.basename(normalized)
        common_paths = [
            os.path.join(project_root, "experiments", "checkpoints", "supervised", filename),
            os.path.join(project_root, "experiments", "checkpoints", "adversarial", filename),
            os.path.join(project_root, "experiments", "checkpoints", "supervised", "best_model.pth"),
            os.path.join(project_root, "experiments", "checkpoints", "adversarial", "best_model.pth"),
        ]
        
        # If the path ends with just a filename, try to find it in common locations
        if os.path.basename(normalized) == normalized or '/' not in normalized.replace('\\', '/'):
            for base in [project_root, os.getcwd()]:
                for subdir in ["experiments/checkpoints/supervised", "experiments/checkpoints/adversarial", "experiments/checkpoints"]:
                    candidate = os.path.join(base, subdir, normalized)
                    if os.path.exists(candidate):
                        return os.path.abspath(candidate)
        
        # Try all common paths
        for path in common_paths:
            if os.path.exists(path):
                return os.path.abspath(path)
        
        return None
        
    def load_model(self):
        """Load trained model"""
        # Resolve model path
        resolved_path = self._resolve_model_path(self.model_path)
        if resolved_path is None:
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Tried to resolve: {self.model_path}\n"
                f"Common locations:\n"
                f"  - experiments/checkpoints/supervised/best_model.pth\n"
                f"  - experiments/checkpoints/adversarial/best_model.pth"
            )
        
        self.logger.info(f"Loading model from {resolved_path}")
        self.model_path = resolved_path
        
        # Load checkpoint
        checkpoint = torch.load(resolved_path, map_location=self.device)
        
        # Determine model type
        model_config = self.config['model']
        
        if model_config['type'] == 'unet':
            self.model = UNet(
                in_channels=model_config['in_channels'],
                out_channels=model_config['out_channels'],
                features=model_config['features'],
                dropout_rate=model_config['dropout_rate'],
                use_attention=model_config['use_attention'],
                use_deep_supervision=model_config['use_deep_supervision']
            )
        elif model_config['type'] == 'unet_light':
            self.model = UNetLight(
                in_channels=model_config['in_channels'],
                out_channels=model_config['out_channels'],
                features=model_config['features']
            )
        elif model_config['type'] == 'multibranch_generator':
            self.model = MultiBranchGenerator(
                in_channels=model_config['in_channels'],
                out_channels=model_config['out_channels'],
                base_channels=model_config['base_channels'],
                num_branches=model_config['num_branches'],
                use_attention=model_config['use_attention'],
                use_wavelet=model_config['use_wavelet']
            )
        elif model_config['type'] == 'generator_light':
            self.model = GeneratorLight(
                in_channels=model_config['in_channels'],
                out_channels=model_config['out_channels'],
                base_channels=model_config['base_channels']
            )
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'generator_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['generator_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"Model loaded successfully")
        
    def setup_data(self):
        """Setup data loaders"""
        self.logger.info("Setting up data loaders...")
        
        # Get augmentation pipeline
        transform = get_augmentation_pipeline('val', self.config['data']['image_size'])
        
        # Create datasets
        self.test_dataset = SARDataset(
            root_dir=self.config['data']['root_dir'],
            split='test',
            transform=transform,
            target_size=(self.config['data']['image_size'], self.config['data']['image_size']),
            filter_method=self.config['data']['filter_method']
        )
        
        # Create data loader
        # Only use pin_memory if CUDA is available
        use_pin_memory = torch.cuda.is_available()
        
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=use_pin_memory
        )
        
        self.logger.info(f"Test samples: {len(self.test_dataset)}")
        
    def setup_metrics(self):
        """Setup metric calculators"""
        # Loss functions
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = PerceptualLoss(self.device)
        self.edge_loss = EdgeLoss()
        
        # LPIPS model for perceptual similarity (offline-safe)
        if LPIPS_AVAILABLE:
            try:
                self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            except Exception as e:
                warnings.warn(
                    f"LPIPS available but could not initialize pretrained network (reason: {e}). "
                    "Proceeding without LPIPS metric."
                )
                self.lpips_model = None
        else:
            self.lpips_model = None
        
    def evaluate_batch(self, sar: torch.Tensor, rgb: torch.Tensor) -> Dict[str, float]:
        """Evaluate a single batch with comprehensive metrics"""
        with torch.no_grad():
            # Generate prediction
            pred = self.model(sar)
            
            # Ensure prediction is in [0, 1] range
            pred = torch.clamp(pred, 0, 1)
            rgb = torch.clamp(rgb, 0, 1)
            
            # Calculate comprehensive metrics
            metrics = {}
            
            # Pixel-wise metrics
            metrics['l1'] = self.l1_loss(pred, rgb).item()
            metrics['l2'] = F.mse_loss(pred, rgb).item()
            
            # Structural similarity
            metrics['ssim'] = 1 - self.ssim_loss(pred, rgb).item()
            
            # Perceptual metrics
            metrics['perceptual'] = self.perceptual_loss(pred, rgb).item()
            metrics['edge'] = self.edge_loss(pred, rgb).item()
            
            # Image quality metrics
            metrics['psnr'] = self._calculate_psnr(pred, rgb)
            metrics['ssim_skimage'] = self._calculate_ssim_skimage(pred, rgb)
            
            # LPIPS (Learned Perceptual Image Patch Similarity)
            if self.lpips_model is not None:
                metrics['lpips'] = self._calculate_lpips(pred, rgb)
            else:
                metrics['lpips'] = 0.0
            
            # Color metrics
            metrics['color_consistency'] = self._calculate_color_consistency(pred, rgb)
            metrics['hue_accuracy'] = self._calculate_hue_accuracy(pred, rgb)
            
            return metrics, pred
    
    def _calculate_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate PSNR using skimage implementation"""
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Calculate PSNR for each image in batch
        psnr_values = []
        for i in range(pred_np.shape[0]):
            pred_img = pred_np[i].transpose(1, 2, 0)
            target_img = target_np[i].transpose(1, 2, 0)
            psnr_val = psnr(target_img, pred_img, data_range=1.0)
            psnr_values.append(psnr_val)
        
        return np.mean(psnr_values)
    
    def _calculate_ssim_skimage(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate SSIM using skimage implementation"""
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Calculate SSIM for each image in batch
        ssim_values = []
        for i in range(pred_np.shape[0]):
            pred_img = pred_np[i].transpose(1, 2, 0)
            target_img = target_np[i].transpose(1, 2, 0)
            # Check image size and determine win_size safely
            H, W, C = pred_img.shape
            safe_win_size = min(7, H, W)
            if safe_win_size % 2 == 0:
                safe_win_size = max(3, safe_win_size - 1)  # Ensure odd and >=3

            if safe_win_size < 3 or H < 3 or W < 3:
                self.logger.warning(
                    f"Image {i} size {H}x{W} too small for SSIM; setting SSIM to 0.0"
                )
                ssim_values.append(0.0)
                continue

            # Handle skimage API differences (multichannel vs channel_axis)
            try:
                ssim_val = ssim(
                    target_img,
                    pred_img,
                    data_range=1.0,
                    win_size=safe_win_size,
                    channel_axis=2,
                )
            except TypeError:
                ssim_val = ssim(
                    target_img,
                    pred_img,
                    data_range=1.0,
                    win_size=safe_win_size,
                    multichannel=True,
                )
            ssim_values.append(ssim_val)

        # Handle the case where all samples were too small
        if not ssim_values:
            return 0.0
        
        return np.mean(ssim_values)
    
    def _calculate_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate LPIPS using the LPIPS model"""
        if self.lpips_model is None:
            return 0.0
        
        # Ensure tensors are in [-1, 1] range for LPIPS
        pred_lpips = pred * 2.0 - 1.0
        target_lpips = target * 2.0 - 1.0
        
        lpips_val = self.lpips_model(pred_lpips, target_lpips)
        return lpips_val.mean().item()
    
    def _calculate_color_consistency(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate color consistency between prediction and target"""
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Convert to grayscale for correlation
        pred_gray = np.mean(pred_np, axis=1)  # Average across channels
        target_gray = np.mean(target_np, axis=1)
        
        # Calculate correlation for each image
        correlations = []
        for i in range(pred_gray.shape[0]):
            corr = np.corrcoef(pred_gray[i].flatten(), target_gray[i].flatten())[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_hue_accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate hue accuracy between prediction and target"""
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Convert RGB to HSV
        def rgb_to_hsv(rgb):
            rgb = np.clip(rgb, 0, 1)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            return hsv[:, :, 0]  # Hue channel
        
        hue_accuracies = []
        for i in range(pred_np.shape[0]):
            pred_hsv = rgb_to_hsv(pred_np[i].transpose(1, 2, 0))
            target_hsv = rgb_to_hsv(target_np[i].transpose(1, 2, 0))
            
            # Calculate hue difference (circular distance)
            hue_diff = np.abs(pred_hsv - target_hsv)
            hue_diff = np.minimum(hue_diff, 360 - hue_diff)  # Circular distance
            hue_accuracy = 1.0 - np.mean(hue_diff) / 180.0  # Normalize to [0, 1]
            hue_accuracies.append(hue_accuracy)
        
        return np.mean(hue_accuracies)
    
    def evaluate_dataset(self) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """Evaluate entire dataset"""
        self.logger.info("Evaluating dataset...")
        
        all_metrics = []
        epoch_metrics = {
            'l1': 0.0, 'ssim': 0.0, 'perceptual': 0.0, 
            'edge': 0.0, 'psnr': 0.0, 'lpips': 0.0
        }
        
        pbar = tqdm(self.test_loader, desc="Evaluation")
        
        for batch_idx, (sar, rgb) in enumerate(pbar):
            sar, rgb = sar.to(self.device), rgb.to(self.device)
            
            # Evaluate batch
            batch_metrics, pred = self.evaluate_batch(sar, rgb)
            all_metrics.append(batch_metrics)
            
            # Update epoch metrics
            for key in epoch_metrics:
                epoch_metrics[key] += batch_metrics[key]
            
            # Update progress bar
            pbar.set_postfix({
                'PSNR': f"{batch_metrics['psnr']:.2f}",
                'SSIM': f"{batch_metrics['ssim']:.4f}"
            })
            
            # Save sample images
            if batch_idx < self.config['evaluation']['num_samples']:
                self.save_sample_images(sar, rgb, pred, batch_idx)
        
        # Average metrics
        num_batches = len(self.test_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics, all_metrics
    
    def save_sample_images(self, sar: torch.Tensor, rgb: torch.Tensor, 
                          pred: torch.Tensor, batch_idx: int):
        """Save sample images for visualization"""
        # Convert to numpy
        sar_np = sar[0].cpu().numpy().squeeze()
        rgb_np = rgb[0].cpu().numpy().transpose(1, 2, 0)
        pred_np = pred[0].cpu().numpy().transpose(1, 2, 0)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # SAR image
        axes[0].imshow(sar_np, cmap='gray')
        axes[0].set_title('SAR Input')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(rgb_np)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(pred_np)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(self.output_dir / 'images' / f'sample_{batch_idx:03d}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_metrics_plot(self, all_metrics: List[Dict[str, float]]):
        """Create metrics visualization"""
        # Convert to DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics = ['l1', 'ssim', 'perceptual', 'edge', 'psnr', 'lpips']
        titles = ['L1 Loss', 'SSIM', 'Perceptual Loss', 'Edge Loss', 'PSNR', 'LPIPS']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            if metric in df.columns:
                axes[i].hist(df[metric], bins=50, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{title} Distribution')
                axes[i].set_xlabel(title)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'metrics_distribution.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create correlation matrix
        plt.figure(figsize=(10, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f')
        plt.title('Metrics Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'correlation_matrix.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_summary_report(self, epoch_metrics: Dict[str, float], 
                            all_metrics: List[Dict[str, float]]):
        """Create comprehensive summary report"""
        # Calculate statistics
        df = pd.DataFrame(all_metrics)
        stats = df.describe()
        
        # Create report
        report = f"""
# SAR Image Colorization Evaluation Report

## Model Information
- Model Type: {self.config['model']['type']}
- Checkpoint: {self.model_path}
- Test Samples: {len(self.test_dataset)}

## Average Metrics
- L1 Loss: {epoch_metrics['l1']:.6f}
- SSIM: {epoch_metrics['ssim']:.6f}
- Perceptual Loss: {epoch_metrics['perceptual']:.6f}
- Edge Loss: {epoch_metrics['edge']:.6f}
- PSNR: {epoch_metrics['psnr']:.2f} dB
- LPIPS: {epoch_metrics['lpips']:.6f}

## Statistics
{stats.to_string()}

## Model Performance
- Best PSNR: {df['psnr'].max():.2f} dB
- Worst PSNR: {df['psnr'].min():.2f} dB
- Best SSIM: {df['ssim'].max():.6f}
- Worst SSIM: {df['ssim'].min():.6f}

## Files Generated
- Sample images: {self.output_dir / 'images'}
- Metrics plots: {self.output_dir / 'plots'}
- Detailed metrics: {self.output_dir / 'metrics' / 'detailed_metrics.csv'}
"""
        
        # Save report
        with open(self.output_dir / 'evaluation_report.md', 'w') as f:
            f.write(report)
        
        # Save detailed metrics
        df.to_csv(self.output_dir / 'metrics' / 'detailed_metrics.csv', index=False)
        
        # Save summary
        summary = {
            'model_type': self.config['model']['type'],
            'checkpoint': self.model_path,
            'test_samples': len(self.test_dataset),
            'average_metrics': epoch_metrics,
            'statistics': stats.to_dict()
        }
        
        import json
        with open(self.output_dir / 'metrics' / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Evaluation report saved to {self.output_dir}")
    
    def run_evaluation(self):
        """Run complete evaluation"""
        self.logger.info("Starting evaluation...")
        
        # Evaluate dataset
        epoch_metrics, all_metrics = self.evaluate_dataset()
        
        # Print results
        self.logger.info("Evaluation Results:")
        self.logger.info(f"L1 Loss: {epoch_metrics['l1']:.6f}")
        self.logger.info(f"SSIM: {epoch_metrics['ssim']:.6f}")
        self.logger.info(f"Perceptual Loss: {epoch_metrics['perceptual']:.6f}")
        self.logger.info(f"Edge Loss: {epoch_metrics['edge']:.6f}")
        self.logger.info(f"PSNR: {epoch_metrics['psnr']:.2f} dB")
        self.logger.info(f"LPIPS: {epoch_metrics['lpips']:.6f}")
        
        # Create visualizations
        self.create_metrics_plot(all_metrics)
        
        # Create summary report
        self.create_summary_report(epoch_metrics, all_metrics)
        
        self.logger.info("Evaluation completed!")


class ClassificationEvaluator:
    """Classification evaluation for MSTAR/SEN12-FLOOD tasks"""
    
    def __init__(self, config: dict, model_path: str, device: str = 'auto'):
        self.config = config
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != 'cpu' else "cpu")
        
        # Setup logging
        self.setup_logging()
        
        # Load model
        self.load_model()
        
        # Setup data
        self.setup_data()
    
    def setup_logging(self):
        """Setup logging"""
        log_dir = Path(self.config['experiment']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'classification_evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """Load classification model"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model = checkpoint['model']
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def setup_data(self):
        """Setup classification data"""
        # This would be implemented based on specific classification task
        # For now, we'll create a placeholder
        self.test_loader = None
        self.class_names = []
    
    def evaluate_classification(self, predictions: np.ndarray, targets: np.ndarray, 
                              class_names: List[str] = None) -> Dict[str, float]:
        """Evaluate classification performance"""
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(np.unique(targets)))]
        
        # Calculate basic metrics
        accuracy = np.mean(predictions == targets)
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Classification report
        report = classification_report(targets, predictions, target_names=class_names, output_dict=True)
        
        # ROC-AUC for binary classification
        roc_auc = None
        if len(np.unique(targets)) == 2:
            try:
                roc_auc = roc_auc_score(targets, predictions)
            except:
                roc_auc = None
        
        # Precision-Recall AUC
        pr_auc = None
        if len(np.unique(targets)) == 2:
            try:
                pr_auc = average_precision_score(targets, predictions)
            except:
                pr_auc = None
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'class_names': class_names
        }
    
    def plot_classification_results(self, results: Dict[str, float], save_path: str = None):
        """Plot classification evaluation results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Classification Evaluation Results', fontsize=16, fontweight='bold')
        
        # Confusion Matrix
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=results['class_names'],
                   yticklabels=results['class_names'], ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # ROC Curve (if binary classification)
        if results['roc_auc'] is not None:
            fpr, tpr, _ = roc_curve(results['targets'], results['predictions'])
            axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {results["roc_auc"]:.3f})')
            axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'ROC Curve\n(Not available for multi-class)', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('ROC Curve')
        
        # Precision-Recall Curve (if binary classification)
        if results['pr_auc'] is not None:
            precision, recall, _ = precision_recall_curve(results['targets'], results['predictions'])
            axes[1, 0].plot(recall, precision, label=f'PR Curve (AUC = {results["pr_auc"]:.3f})')
            axes[1, 0].set_xlabel('Recall')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_title('Precision-Recall Curve')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'PR Curve\n(Not available for multi-class)', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Precision-Recall Curve')
        
        # Metrics Summary
        axes[1, 1].axis('off')
        metrics_text = f"""
        Classification Metrics Summary
        
        Overall Accuracy: {results['accuracy']:.4f}
        
        Per-Class Metrics:
        """
        
        for class_name in results['class_names']:
            if class_name in results['classification_report']:
                metrics = results['classification_report'][class_name]
                metrics_text += f"""
        {class_name}:
          Precision: {metrics['precision']:.4f}
          Recall: {metrics['recall']:.4f}
          F1-Score: {metrics['f1-score']:.4f}
        """
        
        if results['roc_auc'] is not None:
            metrics_text += f"\nROC-AUC: {results['roc_auc']:.4f}"
        if results['pr_auc'] is not None:
            metrics_text += f"\nPR-AUC: {results['pr_auc']:.4f}"
        
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def run_evaluation(self, predictions: np.ndarray, targets: np.ndarray, 
                      class_names: List[str] = None):
        """Run complete classification evaluation"""
        
        self.logger.info("Starting classification evaluation...")
        
        # Evaluate classification
        results = self.evaluate_classification(predictions, targets, class_names)
        
        # Print results
        self.logger.info("Classification Results:")
        self.logger.info(f"Accuracy: {results['accuracy']:.4f}")
        
        if results['roc_auc'] is not None:
            self.logger.info(f"ROC-AUC: {results['roc_auc']:.4f}")
        if results['pr_auc'] is not None:
            self.logger.info(f"PR-AUC: {results['pr_auc']:.4f}")
        
        # Plot results
        self.plot_classification_results(results)
        
        self.logger.info("Classification evaluation completed!")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="SAR Image Colorization Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default='auto', help="Device to use (auto/cpu/cuda)")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    seed_everything(config['experiment']['seed'])
    
    # Create evaluator
    evaluator = ModelEvaluator(config, args.model, args.device)
    
    # Run evaluation
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
