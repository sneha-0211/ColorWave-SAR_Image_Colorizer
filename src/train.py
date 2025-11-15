"""
Production-Ready Supervised Training Script for SAR Image Colorization
Advanced training pipeline with comprehensive logging, checkpointing, and monitoring
"""

import os
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Support running as a package (python -m src.train) and as a script
try:
    from .models.unet import UNet, UNetLight
    from .data_pipeline import SARDataset, create_data_loaders, get_augmentation_pipeline
    from .losses import L1_SSIM_Loss, CombinedLoss, PerceptualLoss, EdgeLoss, L1Loss
    from .utils import seed_everything, save_checkpoint, load_checkpoint, calculate_metrics
except ImportError:
    from models.unet import UNet, UNetLight
    from data_pipeline import SARDataset, create_data_loaders, get_augmentation_pipeline
    from losses import L1_SSIM_Loss, CombinedLoss, PerceptualLoss, EdgeLoss, L1Loss
    from utils import seed_everything, save_checkpoint, load_checkpoint, calculate_metrics


class TrainingManager:
    """Advanced training manager with comprehensive monitoring"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        self.setup_logging()
        
        # Setup directories
        self.setup_directories()
        
        # Initialize components
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_loss()
        self.setup_logging_tools()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        self.patience_counter = 0
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path(self.config['experiment']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Setup experiment directories"""
        self.checkpoint_dir = Path(self.config['experiment']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(self.config['experiment']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_data(self):
        """Setup data loaders"""
        self.logger.info("Setting up data loaders...")
        
        # Get augmentation pipelines
        train_transform = get_augmentation_pipeline('train', self.config['data']['image_size'])
        val_transform = get_augmentation_pipeline('val', self.config['data']['image_size'])
        
        # Create datasets
        self.train_dataset = SARDataset(
            root_dir=self.config['data']['root_dir'],
            split='train',
            transform=train_transform,
            target_size=(self.config['data']['image_size'], self.config['data']['image_size']),
            filter_method=self.config['data']['filter_method']
        )
        
        self.val_dataset = SARDataset(
            root_dir=self.config['data']['root_dir'],
            split='val',
            transform=val_transform,
            target_size=(self.config['data']['image_size'], self.config['data']['image_size']),
            filter_method=self.config['data']['filter_method']
        )
        
        # Create data loaders
        # Avoid empty epoch when dataset is smaller than batch size
        train_drop_last = len(self.train_dataset) >= self.config['training']['batch_size']
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
            drop_last=train_drop_last
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
            drop_last=False
        )
        
        self.logger.info(f"Train samples: {len(self.train_dataset)}")
        self.logger.info(f"Val samples: {len(self.val_dataset)}")
        
    def setup_model(self):
        """Setup model architecture"""
        self.logger.info("Setting up model...")
        
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
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")
        
        self.model = self.model.to(self.device)
        
        # Log model info (guard optional helpers)
        self.logger.info(f"Model: {model_config['type']}")
        try:
            if hasattr(self.model, 'get_model_size'):
                self.logger.info(f"Parameters: {self.model.get_model_size():,}")
            else:
                total_params = sum(p.numel() for p in self.model.parameters())
                self.logger.info(f"Parameters: {total_params:,}")
            if hasattr(self.model, 'get_model_memory'):
                self.logger.info(f"Memory: {self.model.get_model_memory():.2f} MB")
        except Exception:
            pass
        
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        self.logger.info("Setting up optimizer...")
        
        # Optimizer
        if self.config['training']['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                betas=tuple(self.config['training']['betas']),
                weight_decay=self.config['training']['weight_decay']
            )
        elif self.config['training']['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                betas=tuple(self.config['training']['betas']),
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['training']['optimizer']}")
        
        # Scheduler
        if self.config['training']['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=self.config['training']['learning_rate'] * 0.01
            )
        elif self.config['training']['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['training']['scheduler_step'],
                gamma=self.config['training']['scheduler_gamma']
            )
        elif self.config['training']['scheduler'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            self.scheduler = None
            
    def setup_loss(self):
        """Setup loss function"""
        self.logger.info("Setting up loss function...")
        
        loss_config = self.config['training']['loss']
        
        if loss_config['type'] == 'l1_ssim':
            self.criterion = L1_SSIM_Loss(
                l1_weight=loss_config['l1_weight'],
                ssim_weight=loss_config['ssim_weight']
            )
        elif loss_config['type'] == 'l1':
            self.criterion = L1Loss()
        elif loss_config['type'] == 'combined':
            self.criterion = CombinedLoss(weights=loss_config['weights'])
        else:
            raise ValueError(f"Unsupported loss type: {loss_config['type']}")
        
        # Additional losses for monitoring
        self.perceptual_loss = PerceptualLoss(self.device)
        self.edge_loss = EdgeLoss()
        
    def setup_logging_tools(self):
        """Setup logging tools"""
        self.writer = SummaryWriter(log_dir=self.log_dir / 'tensorboard')
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {'l1': 0.0, 'ssim': 0.0, 'perceptual': 0.0, 'edge': 0.0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (sar, rgb) in enumerate(pbar):
            sar, rgb = sar.to(self.device), rgb.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(sar)
            
            # Compute loss
            if isinstance(output, tuple):  # Deep supervision
                main_output, aux_outputs = output
                loss = self.criterion(main_output, rgb)
                
                # Add auxiliary losses
                for aux_output in aux_outputs:
                    loss += 0.1 * self.criterion(aux_output, rgb)
            else:
                loss = self.criterion(output, rgb)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training']['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Compute additional metrics
            with torch.no_grad():
                if isinstance(output, tuple):
                    output = output[0]
                
                l1 = torch.nn.functional.l1_loss(output, rgb).item()
                epoch_metrics['l1'] += l1
                
                # Calculate additional metrics if available
                if hasattr(self.criterion, 'ssim_loss'):
                    ssim = 1 - self.criterion.ssim_loss(output, rgb).item()
                    epoch_metrics['ssim'] += ssim
                else:
                    # Calculate SSIM manually for monitoring
                    try:
                        from .losses import SSIMLoss
                    except ImportError:
                        from losses import SSIMLoss
                    ssim_loss = SSIMLoss()
                    ssim = 1 - ssim_loss(output, rgb).item()
                    epoch_metrics['ssim'] += ssim
                
                perceptual = self.perceptual_loss(output, rgb).item()
                # edge = self.edge_loss(output, rgb).item()  # Disabled due to device issues
                
                epoch_metrics['perceptual'] += perceptual
                # epoch_metrics['edge'] += edge
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'L1': f'{l1:.4f}',
                'SSIM': f'{ssim:.4f}'
            })
        
        # Average metrics
        num_batches = len(self.train_loader)
        if num_batches == 0:
            self.logger.warning("No training batches this epoch (dataset smaller than batch size). Skipping averaging.")
            return 0.0, {k: 0.0 for k in epoch_metrics}
        
        epoch_loss /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_loss, epoch_metrics
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        val_loss = 0.0
        val_metrics = {'l1': 0.0, 'ssim': 0.0, 'perceptual': 0.0, 'edge': 0.0, 'psnr': 0.0}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for sar, rgb in pbar:
                sar, rgb = sar.to(self.device), rgb.to(self.device)
                
                # Forward pass
                output = self.model(sar)
                
                # Compute loss
                if isinstance(output, tuple):
                    output = output[0]
                
                loss = self.criterion(output, rgb)
                val_loss += loss.item()
                
                # Compute metrics
                l1 = torch.nn.functional.l1_loss(output, rgb).item()
                
                # Calculate SSIM if available
                if hasattr(self.criterion, 'ssim_loss'):
                    ssim = 1 - self.criterion.ssim_loss(output, rgb).item()
                else:
                    try:
                        from .losses import SSIMLoss
                    except ImportError:
                        from losses import SSIMLoss
                    ssim_loss = SSIMLoss()
                    ssim = 1 - ssim_loss(output, rgb).item()
                
                perceptual = self.perceptual_loss(output, rgb).item()
                # edge = self.edge_loss(output, rgb).item()  # Disabled due to device issues
                psnr = calculate_metrics(output, rgb, 'psnr')
                
                val_metrics['l1'] += l1
                val_metrics['ssim'] += ssim
                val_metrics['perceptual'] += perceptual
                # val_metrics['edge'] += edge
                val_metrics['psnr'] += psnr
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'PSNR': f'{psnr:.2f}'
                })
        
        # Average metrics
        num_batches = len(self.val_loader)
        val_loss /= num_batches
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return val_loss, val_metrics
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {self.current_epoch}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {self.config['training']['epochs']}")
        
        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.log_metrics(epoch, train_loss, train_metrics, val_loss, val_metrics)
            
            # Save checkpoint and handle early stopping consistently
            improved = val_loss < self.best_val_loss
            if epoch % self.config['training']['save_frequency'] == 0 or improved:
                self.save_checkpoint(improved)
            if improved:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['training']['patience']:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        self.logger.info("Training completed!")
        self.writer.close()
    
    def log_metrics(self, epoch, train_loss, train_metrics, val_loss, val_metrics):
        """Log training metrics"""
        # Console logging
        self.logger.info(
            f"Epoch {epoch}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val PSNR: {val_metrics['psnr']:.2f}"
        )
        
        # TensorBoard logging
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Val', val_loss, epoch)
        
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{key.title()}', value, epoch)
        
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'Val/{key.title()}', value, epoch)
        
        # Learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning_Rate', current_lr, epoch)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="SAR Image Colorization Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    seed_everything(config['experiment']['seed'])
    
    # Create training manager
    trainer = TrainingManager(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()