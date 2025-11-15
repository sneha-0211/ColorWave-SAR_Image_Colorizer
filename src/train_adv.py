"""
Production-Ready Adversarial Training Script for SAR Image Colorization
Advanced GAN training with multi-scale discriminators and comprehensive monitoring
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

# Support running as a package and as a script
try:
    from .models.generator_adv import MultiBranchGenerator, GeneratorLight
    from .models.discriminator import PatchDiscriminator, MultiScaleDiscriminator, EnsembleDiscriminator
    from .data_pipeline import SARDataset, create_data_loaders, get_augmentation_pipeline
    from .losses import (
        L1Loss, SSIMLoss, PerceptualLoss, GANLoss, GradientPenaltyLoss,
        CombinedLoss, EdgeLoss, TotalVariationLoss
    )
    from .utils import seed_everything, save_checkpoint, load_checkpoint, calculate_metrics
except ImportError:
    from models.generator_adv import MultiBranchGenerator, GeneratorLight
    from models.discriminator import PatchDiscriminator, MultiScaleDiscriminator, EnsembleDiscriminator
    from data_pipeline import SARDataset, create_data_loaders, get_augmentation_pipeline
    from losses import (
        L1Loss, SSIMLoss, PerceptualLoss, GANLoss, GradientPenaltyLoss,
        CombinedLoss, EdgeLoss, TotalVariationLoss
    )
    from utils import seed_everything, save_checkpoint, load_checkpoint, calculate_metrics


class AdversarialTrainingManager:
    """Advanced adversarial training manager"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        self.setup_logging()
        
        # Setup directories
        self.setup_directories()
        
        # Initialize components
        self.setup_data()
        self.setup_models()
        self.setup_optimizers()
        self.setup_losses()
        self.setup_logging_tools()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path(self.config['experiment']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'adversarial_training.log'),
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
        
    def setup_models(self):
        """Setup generator and discriminator models"""
        self.logger.info("Setting up models...")
        
        # Generator
        gen_config = self.config['model']['generator']
        if gen_config['type'] == 'multibranch':
            self.generator = MultiBranchGenerator(
                in_channels=gen_config['in_channels'],
                out_channels=gen_config['out_channels'],
                base_channels=gen_config['base_channels'],
                num_branches=gen_config['num_branches'],
                use_attention=gen_config['use_attention'],
                use_wavelet=gen_config['use_wavelet']
            )
        elif gen_config['type'] == 'light':
            self.generator = GeneratorLight(
                in_channels=gen_config['in_channels'],
                out_channels=gen_config['out_channels'],
                base_channels=gen_config['base_channels']
            )
        else:
            raise ValueError(f"Unsupported generator type: {gen_config['type']}")
        
        # Discriminator
        disc_config = self.config['model']['discriminator']
        if disc_config['type'] == 'patch':
            self.discriminator = PatchDiscriminator(
                in_channels=disc_config['in_channels'],
                base_channels=disc_config['base_channels'],
                num_layers=disc_config['num_layers'],
                use_sn=disc_config['use_sn']
            )
        elif disc_config['type'] == 'multiscale':
            self.discriminator = MultiScaleDiscriminator(
                in_channels=disc_config['in_channels'],
                base_channels=disc_config['base_channels'],
                num_scales=disc_config['num_scales'],
                use_sn=disc_config['use_sn']
            )
        elif disc_config['type'] == 'ensemble':
            self.discriminator = EnsembleDiscriminator(
                in_channels=disc_config['in_channels'],
                base_channels=disc_config['base_channels'],
                num_discriminators=disc_config['num_discriminators']
            )
        else:
            raise ValueError(f"Unsupported discriminator type: {disc_config['type']}")
        
        # Move to device
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        
        # Log model info
        self.logger.info(f"Generator: {gen_config['type']}")
        self.logger.info(f"Generator parameters: {self.generator.get_model_size():,}")
        self.logger.info(f"Discriminator: {disc_config['type']}")
        
    def setup_optimizers(self):
        """Setup optimizers and schedulers"""
        self.logger.info("Setting up optimizers...")
        
        # Generator optimizer
        gen_opt_config = self.config['training']['generator_optimizer']
        if gen_opt_config['type'] == 'adam':
            self.optimizer_g = optim.Adam(
                self.generator.parameters(),
                lr=gen_opt_config['learning_rate'],
                betas=tuple(gen_opt_config['betas']),
                weight_decay=gen_opt_config['weight_decay']
            )
        elif gen_opt_config['type'] == 'adamw':
            self.optimizer_g = optim.AdamW(
                self.generator.parameters(),
                lr=gen_opt_config['learning_rate'],
                betas=tuple(gen_opt_config['betas']),
                weight_decay=gen_opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported generator optimizer: {gen_opt_config['type']}")
        
        # Discriminator optimizer
        disc_opt_config = self.config['training']['discriminator_optimizer']
        if disc_opt_config['type'] == 'adam':
            self.optimizer_d = optim.Adam(
                self.discriminator.parameters(),
                lr=disc_opt_config['learning_rate'],
                betas=tuple(disc_opt_config['betas']),
                weight_decay=disc_opt_config['weight_decay']
            )
        elif disc_opt_config['type'] == 'adamw':
            self.optimizer_d = optim.AdamW(
                self.discriminator.parameters(),
                lr=disc_opt_config['learning_rate'],
                betas=tuple(disc_opt_config['betas']),
                weight_decay=disc_opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported discriminator optimizer: {disc_opt_config['type']}")
        
        # Schedulers
        if self.config['training']['scheduler'] == 'cosine':
            self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_g,
                T_max=self.config['training']['epochs'],
                eta_min=gen_opt_config['learning_rate'] * 0.01
            )
            self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_d,
                T_max=self.config['training']['epochs'],
                eta_min=disc_opt_config['learning_rate'] * 0.01
            )
        else:
            self.scheduler_g = None
            self.scheduler_d = None
            
    def setup_losses(self):
        """Setup loss functions"""
        self.logger.info("Setting up loss functions...")
        
        # In config, adversarial loss settings are under top-level 'loss'
        loss_config = self.config['loss']
        
        # Individual losses
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = PerceptualLoss(self.device)
        self.edge_loss = EdgeLoss()
        self.tv_loss = TotalVariationLoss()
        
        # GAN losses
        self.gan_loss = GANLoss(
            gan_mode=loss_config['gan_mode'],
            target_real_label=1.0,
            target_fake_label=0.0
        )
        
        # Gradient penalty for WGAN-GP
        if loss_config['gan_mode'] == 'wgangp':
            self.gp_loss = GradientPenaltyLoss(lambda_gp=loss_config['lambda_gp'])
        else:
            self.gp_loss = None
        
        # Loss weights
        self.lambda_l1 = loss_config['lambda_l1']
        self.lambda_ssim = loss_config['lambda_ssim']
        self.lambda_perceptual = loss_config['lambda_perceptual']
        self.lambda_gan = loss_config['lambda_gan']
        self.lambda_edge = loss_config['lambda_edge']
        self.lambda_tv = loss_config['lambda_tv']
        
    def setup_logging_tools(self):
        """Setup logging tools"""
        self.writer = SummaryWriter(log_dir=self.log_dir / 'tensorboard')

    def _gan_loss_from_outputs(self, outputs, target_is_real: bool):
        """Compute GAN loss supporting tensor or list of tensors (multi-scale)."""
        if isinstance(outputs, (list, tuple)):
            total = 0
            for out in outputs:
                total += self.gan_loss(out, target_is_real)
            return total / len(outputs)
        return self.gan_loss(outputs, target_is_real)

    def _gp_wrapper(self, real_input, fake_input):
        """Gradient penalty that works with multi-output discriminators."""
        if self.gp_loss is None:
            return 0.0
        def disc_main(x):
            out = self.discriminator(x)
            return out[0] if isinstance(out, (list, tuple)) else out
        return self.gp_loss(disc_main, real_input, fake_input)
        
    def train_epoch(self):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = {
            'g_loss': 0.0, 'd_loss': 0.0,
            'l1': 0.0, 'ssim': 0.0, 'perceptual': 0.0, 'edge': 0.0,
            'gan_g': 0.0, 'gan_d': 0.0, 'gp': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (sar, rgb) in enumerate(pbar):
            sar, rgb = sar.to(self.device), rgb.to(self.device)
            
            # Generate fake images
            fake_rgb = self.generator(sar)
            
            # Train Discriminator
            d_loss = self.train_discriminator(sar, rgb, fake_rgb)
            
            # Train Generator
            g_loss = self.train_generator(sar, rgb, fake_rgb)
            
            # Update metrics
            epoch_metrics['g_loss'] += g_loss['total']
            epoch_metrics['d_loss'] += d_loss['total']
            epoch_metrics['l1'] += g_loss['l1']
            epoch_metrics['ssim'] += g_loss['ssim']
            epoch_metrics['perceptual'] += g_loss['perceptual']
            epoch_metrics['edge'] += g_loss['edge']
            epoch_metrics['gan_g'] += g_loss['gan']
            epoch_metrics['gan_d'] += d_loss['gan']
            if 'gp' in d_loss:
                epoch_metrics['gp'] += d_loss['gp']
            
            # Update progress bar
            pbar.set_postfix({
                'G_Loss': f"{g_loss['total']:.4f}",
                'D_Loss': f"{d_loss['total']:.4f}",
                'L1': f"{g_loss['l1']:.4f}"
            })
        
        # Average metrics
        num_batches = len(self.train_loader)
        if num_batches == 0:
            self.logger.warning("No training batches this epoch (dataset smaller than batch size).")
            return epoch_metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def train_discriminator(self, sar, rgb, fake_rgb):
        """Train discriminator"""
        self.optimizer_d.zero_grad()
        
        # Real samples
        real_input = torch.cat([sar, rgb], dim=1)
        real_output = self.discriminator(real_input)
        real_loss = self._gan_loss_from_outputs(real_output, True)
        
        # Fake samples
        fake_input = torch.cat([sar, fake_rgb.detach()], dim=1)
        fake_output = self.discriminator(fake_input)
        fake_loss = self._gan_loss_from_outputs(fake_output, False)
        
        # Total discriminator loss
        d_loss = 0.5 * (real_loss + fake_loss)
        
        # Gradient penalty for WGAN-GP
        if self.gp_loss is not None:
            gp = self._gp_wrapper(real_input, fake_input)
            d_loss += gp
        else:
            gp = 0.0
        
        d_loss.backward()
        self.optimizer_d.step()
        
        return {
            'total': d_loss.item(),
            'gan': d_loss.item(),
            'gp': gp.item() if isinstance(gp, torch.Tensor) else float(gp)
        }
    
    def train_generator(self, sar, rgb, fake_rgb):
        """Train generator"""
        self.optimizer_g.zero_grad()
        
        # Adversarial loss
        fake_input = torch.cat([sar, fake_rgb], dim=1)
        fake_output = self.discriminator(fake_input)
        gan_loss = self._gan_loss_from_outputs(fake_output, True)
        
        # Reconstruction losses
        l1 = self.l1_loss(fake_rgb, rgb)
        ssim = self.ssim_loss(fake_rgb, rgb)
        perceptual = self.perceptual_loss(fake_rgb, rgb)
        edge = self.edge_loss(fake_rgb, rgb)
        tv = self.tv_loss(fake_rgb)
        
        # Total generator loss
        g_loss = (
            self.lambda_l1 * l1 +
            self.lambda_ssim * ssim +
            self.lambda_perceptual * perceptual +
            self.lambda_gan * gan_loss +
            self.lambda_edge * edge +
            self.lambda_tv * tv
        )
        
        g_loss.backward()
        
        # Gradient clipping
        if self.config['training']['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), 
                self.config['training']['grad_clip']
            )
        
        self.optimizer_g.step()
        
        return {
            'total': g_loss.item(),
            'l1': l1.item(),
            'ssim': ssim.item(),
            'perceptual': perceptual.item(),
            'edge': edge.item(),
            'tv': tv.item(),
            'gan': gan_loss.item()
        }
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.generator.eval()
        self.discriminator.eval()
        
        val_metrics = {
            'g_loss': 0.0, 'd_loss': 0.0,
            'l1': 0.0, 'ssim': 0.0, 'perceptual': 0.0, 'edge': 0.0,
            'psnr': 0.0
        }
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for sar, rgb in pbar:
                sar, rgb = sar.to(self.device), rgb.to(self.device)
                
                # Generate fake images
                fake_rgb = self.generator(sar)
                
                # Compute losses
                l1 = self.l1_loss(fake_rgb, rgb)
                ssim = self.ssim_loss(fake_rgb, rgb)
                perceptual = self.perceptual_loss(fake_rgb, rgb)
                edge = self.edge_loss(fake_rgb, rgb)
                
                # Adversarial loss
                fake_input = torch.cat([sar, fake_rgb], dim=1)
                fake_output = self.discriminator(fake_input)
                gan_loss = self._gan_loss_from_outputs(fake_output, True)
                
                # Total losses
                g_loss = (
                    self.lambda_l1 * l1 +
                    self.lambda_ssim * ssim +
                    self.lambda_perceptual * perceptual +
                    self.lambda_gan * gan_loss +
                    self.lambda_edge * edge
                )
                
                # Discriminator loss
                real_input = torch.cat([sar, rgb], dim=1)
                real_output = self.discriminator(real_input)
                real_loss = self._gan_loss_from_outputs(real_output, True)
                fake_loss = self._gan_loss_from_outputs(fake_output, False)
                d_loss = 0.5 * (real_loss + fake_loss)
                
                # PSNR
                psnr = calculate_metrics(fake_rgb, rgb, 'psnr')
                
                # Update metrics
                val_metrics['g_loss'] += g_loss.item()
                val_metrics['d_loss'] += d_loss.item()
                val_metrics['l1'] += l1.item()
                val_metrics['ssim'] += ssim.item()
                val_metrics['perceptual'] += perceptual.item()
                val_metrics['edge'] += edge.item()
                val_metrics['psnr'] += psnr
                
                pbar.set_postfix({
                    'G_Loss': f"{g_loss.item():.4f}",
                    'PSNR': f"{psnr:.2f}"
                })
        
        # Average metrics
        num_batches = len(self.val_loader)
        for key in val_metrics:
            val_metrics[key] /= max(num_batches, 1)
        
        return val_metrics
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoints"""
        checkpoint = {
            'epoch': self.current_epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict() if self.scheduler_g else None,
            'scheduler_d_state_dict': self.scheduler_d.state_dict() if self.scheduler_d else None,
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
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        
        if self.scheduler_g and checkpoint['scheduler_g_state_dict']:
            self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        if self.scheduler_d and checkpoint['scheduler_d_state_dict']:
            self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting adversarial training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {self.config['training']['epochs']}")
        
        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update schedulers
            if self.scheduler_g:
                self.scheduler_g.step()
            if self.scheduler_d:
                self.scheduler_d.step()
            
            # Log metrics
            self.log_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            is_best = val_metrics['g_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['g_loss']
            
            if epoch % self.config['training']['save_frequency'] == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Early stopping
            if hasattr(self, 'patience_counter'):
                if val_metrics['g_loss'] < self.best_val_loss:
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config['training']['patience']:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break
        
        self.logger.info("Adversarial training completed!")
        self.writer.close()
    
    def log_metrics(self, epoch, train_metrics, val_metrics):
        """Log training metrics"""
        # Console logging
        self.logger.info(
            f"Epoch {epoch}: "
            f"Train G: {train_metrics['g_loss']:.4f}, "
            f"Train D: {train_metrics['d_loss']:.4f}, "
            f"Val G: {val_metrics['g_loss']:.4f}, "
            f"Val PSNR: {val_metrics['psnr']:.2f}"
        )
        
        # TensorBoard logging
        self.writer.add_scalar('Loss/Generator/Train', train_metrics['g_loss'], epoch)
        self.writer.add_scalar('Loss/Discriminator/Train', train_metrics['d_loss'], epoch)
        self.writer.add_scalar('Loss/Generator/Val', val_metrics['g_loss'], epoch)
        self.writer.add_scalar('Loss/Discriminator/Val', val_metrics['d_loss'], epoch)
        
        # Individual losses
        for key in ['l1', 'ssim', 'perceptual', 'edge', 'gan_g', 'gan_d']:
            if key in train_metrics:
                self.writer.add_scalar(f'Train/{key.title()}', train_metrics[key], epoch)
            if key in val_metrics:
                self.writer.add_scalar(f'Val/{key.title()}', val_metrics[key], epoch)
        
        # Learning rates
        current_lr_g = self.optimizer_g.param_groups[0]['lr']
        current_lr_d = self.optimizer_d.param_groups[0]['lr']
        self.writer.add_scalar('Learning_Rate/Generator', current_lr_g, epoch)
        self.writer.add_scalar('Learning_Rate/Discriminator', current_lr_d, epoch)


def main():
    """Main adversarial training function"""
    parser = argparse.ArgumentParser(description="SAR Image Colorization Adversarial Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    seed_everything(config['experiment']['seed'])

    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Create training manager
    trainer = AdversarialTrainingManager(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()