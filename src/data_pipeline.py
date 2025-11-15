"""
Production-Ready Data Pipeline for SAR Image Colorization
Comprehensive data loading, preprocessing, and augmentation
"""

import os
import argparse
import shutil
import json
import random
import re
from glob import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
from scipy.ndimage import uniform_filter
import cv2
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


@dataclass
class DatasetConfig:
    """Configuration for dataset processing"""
    name: str
    sar_path: str
    optical_path: Optional[str] = None
    tile_size: int = 256
    overlap: float = 0.25
    filter_method: str = 'lee'
    normalization: str = 'robust'
    augmentation: bool = True


class SARImageProcessor:
    """Advanced SAR image processing with multiple filtering options"""
    
    @staticmethod
    def lee_filter(img: np.ndarray, size: int = 7, eps: float = 1e-8) -> np.ndarray:
        """Lee filter for SAR speckle reduction"""
        img = img.astype(np.float32)
        mean = uniform_filter(img, size)
        mean_sq = uniform_filter(img * img, size)
        var = mean_sq - mean * mean
        overall_var = np.maximum(var, eps)
        window_var = uniform_filter((img - mean) ** 2, size)
        w = window_var / (overall_var + eps)
        result = mean + w * (img - mean)
        return result.astype(np.float32)

    @staticmethod
    def median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
        """Median filter for noise reduction"""
        scaled = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        out = cv2.medianBlur(scaled, ksize)
        rev = out.astype(np.float32) / 255.0
        return rev * (img.max() - img.min()) + img.min()

    @staticmethod
    def nl_means_filter(img: np.ndarray, h: float = 10.0) -> np.ndarray:
        """Non-local means denoising"""
        scaled = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        den = cv2.fastNlMeansDenoising(scaled, None, h, 7, 21)
        den = den.astype(np.float32) / 255.0
        return den * (img.max() - img.min()) + img.min()

    @staticmethod
    def robust_normalize(img: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.0, 
                        eps: float = 1e-8) -> np.ndarray:
        """Robust normalization using percentiles"""
        if img.ndim == 3:
            out = np.zeros_like(img, dtype=np.float32)
            for c in range(img.shape[0]):
                channel = img[c]
                lo = np.percentile(channel, lower_pct)
                hi = np.percentile(channel, upper_pct)
                out[c] = np.clip((channel - lo) / max((hi - lo), eps), 0.0, 1.0)
            return out.astype(np.float32)
        else:
            lo = np.percentile(img, lower_pct)
            hi = np.percentile(img, upper_pct)
            out = np.clip((img - lo) / max((hi - lo), eps), 0.0, 1.0)
            return out.astype(np.float32)

    @staticmethod
    def z_score_normalize(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Z-score normalization"""
        if img.ndim == 3:
            out = np.zeros_like(img, dtype=np.float32)
            for c in range(img.shape[0]):
                channel = img[c]
                mean = np.mean(channel)
                std = np.std(channel)
                out[c] = (channel - mean) / max(std, eps)
            return out.astype(np.float32)
        else:
            mean = np.mean(img)
            std = np.std(img)
            return (img - mean) / max(std, eps).astype(np.float32)


class GeoTIFFProcessor:
    """GeoTIFF processing with metadata preservation"""
    
    @staticmethod
    def align_to_reference(src_path: str, ref_path: str, dst_path: str, 
                          resampling=Resampling.bilinear):
        """Align source image to reference image"""
        with rasterio.open(ref_path) as ref:
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_width = ref.width
            ref_height = ref.height
        
        with rasterio.open(src_path) as src:
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': ref_crs,
                'transform': ref_transform,
                'width': ref_width,
                'height': ref_height
            })
            
            with rasterio.open(dst_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=resampling
                    )
    
    @staticmethod
    def tile_and_save(src_path: str, dst_dir: str, tile_size: int = 256, 
                     overlap: float = 0.0, skip_partial: bool = True):
        """Tile large images into smaller patches"""
        os.makedirs(dst_dir, exist_ok=True)
        
        with rasterio.open(src_path) as src:
            step = int(tile_size * (1 - overlap))
            if step <= 0:
                raise ValueError("Invalid overlap")
                
            meta = src.meta.copy()
                
            for x in range(0, src.width, step):
                for y in range(0, src.height, step):
                    win = Window(x, y, tile_size, tile_size)
                        
                    if (x + tile_size > src.width) or (y + tile_size > src.height):
                        if skip_partial:
                            continue
                        
                    try:
                        data = src.read(window=win)
                    except Exception:
                        continue
                        
                    if data.shape[1] != tile_size or data.shape[2] != tile_size:
                        if skip_partial:
                            continue
                        
                    out_meta = meta.copy()
                    out_meta.update({
                        "height": data.shape[1],
                        "width": data.shape[2],
                        "transform": src.window_transform(win),
                        "dtype": rasterio.float32,
                        "driver": "GTiff",  # Explicitly use GTiff driver
                        "compress": "lzw"
                    })
                        
                    out_name = f"{x}_{y}.tif"
                    out_path = os.path.join(dst_dir, out_name)
                        
                    with rasterio.open(out_path, "w", **out_meta) as dst:
                        dst.write(data.astype(np.float32))


class SARDataset(Dataset):
    """Enhanced SAR dataset with comprehensive preprocessing"""
    
    def __init__(self, root_dir: str, split: str = "train", 
                 transform: Optional[A.Compose] = None,
                 target_size: Optional[Tuple[int, int]] = None,
                 normalize: bool = True,
                 filter_method: str = 'none'):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.normalize = normalize
        self.filter_method = filter_method
        
        # Define paths
        self.sar_dir = os.path.join(root_dir, split, "SAR")
        self.opt_dir = os.path.join(root_dir, split, "Optical")
        
        # Get file pairs
        self.pairs = self._get_pairs()
        
        if len(self.pairs) == 0:
            print(f"Warning: No data found in {self.sar_dir} and {self.opt_dir}")
    
    def _get_pairs(self) -> List[Tuple[str, str]]:
        """Get matching SAR and Optical file pairs"""
        sar_files = sorted([f for f in os.listdir(self.sar_dir) 
                           if f.endswith(('.tif', '.png', '.jpg', '.jpeg'))])
        opt_files = sorted([f for f in os.listdir(self.opt_dir) 
                           if f.endswith(('.tif', '.png', '.jpg', '.jpeg'))])
        
        pairs = []
        for sar_file in sar_files:
            base_name = os.path.splitext(sar_file)[0]
            for opt_file in opt_files:
                if os.path.splitext(opt_file)[0] == base_name:
                    pairs.append((sar_file, opt_file))
                    break
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        sar_file, opt_file = self.pairs[idx]
        
        # Load SAR image
        sar_path = os.path.join(self.sar_dir, sar_file)
        file_ext = Path(sar_path).suffix.lower()
        is_raster_format = file_ext in ['.tif', '.tiff']
        
        if is_raster_format:
            try:
                with rasterio.open(sar_path) as src:
                    sar_img = src.read().astype(np.float32)
                    if sar_img.ndim == 3 and sar_img.shape[0] == 1:
                        sar_img = sar_img.squeeze(0)
                    elif sar_img.ndim == 3:
                        sar_img = sar_img[0]  # Take first channel
            except:
                # Fallback to PIL if rasterio fails
                sar_img = np.array(Image.open(sar_path).convert('L')).astype(np.float32)
        else:
            # Use PIL directly for PNG/JPEG files to avoid rasterio warnings
            sar_img = np.array(Image.open(sar_path).convert('L')).astype(np.float32)
        
        # Load Optical image
        opt_path = os.path.join(self.opt_dir, opt_file)
        file_ext = Path(opt_path).suffix.lower()
        is_raster_format = file_ext in ['.tif', '.tiff']
        
        if is_raster_format:
            try:
                with rasterio.open(opt_path) as src:
                    opt_img = src.read().astype(np.float32)
                    if opt_img.ndim == 3 and opt_img.shape[0] == 3:
                        opt_img = opt_img  # Keep RGB channels (C, H, W format)
                    elif opt_img.ndim == 3 and opt_img.shape[2] == 3:
                        opt_img = opt_img.transpose(2, 0, 1)  # Convert (H, W, C) to (C, H, W)
                    else:
                        # Convert to 3 channels if needed
                        if opt_img.ndim == 2:
                            opt_img = np.repeat(opt_img[np.newaxis, :, :], 3, axis=0)
                        else:
                            # If it's already 3D but wrong shape, reshape it
                            if opt_img.ndim == 3:
                                opt_img = opt_img[:3]  # Take first 3 channels
                            else:
                                opt_img = np.repeat(opt_img, 3, axis=0)
            except:
                # Fallback to PIL if rasterio fails
                opt_img = np.array(Image.open(opt_path).convert('RGB')).astype(np.float32)
                opt_img = opt_img.transpose(2, 0, 1)  # Convert (H, W, C) to (C, H, W)
        else:
            # Use PIL directly for PNG/JPEG files to avoid rasterio warnings
            opt_img = np.array(Image.open(opt_path).convert('RGB')).astype(np.float32)
            opt_img = opt_img.transpose(2, 0, 1)  # Convert (H, W, C) to (C, H, W)
        
        # Apply filtering
        if self.filter_method == 'lee':
            sar_img = SARImageProcessor.lee_filter(sar_img)
        elif self.filter_method == 'median':
            sar_img = SARImageProcessor.median_filter(sar_img)
        elif self.filter_method == 'nlmeans':
            sar_img = SARImageProcessor.nl_means_filter(sar_img)
        
        # Normalize
        if self.normalize:
            sar_img = SARImageProcessor.robust_normalize(sar_img)
            opt_img = SARImageProcessor.robust_normalize(opt_img)
        
        # Resize if needed
        if self.target_size:
            sar_img = cv2.resize(sar_img, self.target_size, interpolation=cv2.INTER_LINEAR)
            # For optical image, we need to transpose to (H, W, C) for cv2.resize
            if opt_img.ndim == 3:
                opt_img = opt_img.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                opt_img = cv2.resize(opt_img, self.target_size, interpolation=cv2.INTER_LINEAR)
                opt_img = opt_img.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
            else:
                opt_img = cv2.resize(opt_img, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to torch tensors
        sar_tensor = torch.from_numpy(sar_img).unsqueeze(0)  # Add channel dimension
        opt_tensor = torch.from_numpy(opt_img)  # Should be (C, H, W) format
        
        
        # Apply transforms
        if self.transform:
            # Convert to numpy for albumentations
            sar_np = sar_tensor.squeeze(0).numpy()
            opt_np = opt_tensor.permute(1, 2, 0).numpy()
            
            # Apply augmentation
            augmented = self.transform(image=sar_np, mask=opt_np)
            
            # Handle tensor conversion properly
            if isinstance(augmented['image'], torch.Tensor):
                # Remove extra dimension if present
                if augmented['image'].dim() == 3:
                    sar_tensor = augmented['image'].unsqueeze(0)
                else:
                    sar_tensor = augmented['image']
            else:
                sar_tensor = torch.from_numpy(augmented['image']).unsqueeze(0)
                
            if isinstance(augmented['mask'], torch.Tensor):
                opt_tensor = augmented['mask'].permute(2, 0, 1)
            else:
                opt_tensor = torch.from_numpy(augmented['mask']).permute(2, 0, 1)
        
        return sar_tensor, opt_tensor


def get_augmentation_pipeline(mode: str = 'train', image_size: int = 256) -> A.Compose:
    """Get augmentation pipeline for training/validation"""
    
    if mode == 'train':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            # Use default variance limits to support a wide range of albumentations versions
            A.GaussNoise(p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            A.Normalize(
                mean=[0.5],
                std=[0.5],
                max_pixel_value=1.0
            )
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.5],
                std=[0.5],
                max_pixel_value=1.0
            )
        ])


def create_data_loaders(root_dir: str, batch_size: int = 8, num_workers: int = 4,
                       image_size: int = 256, filter_method: str = 'none') -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training and validation"""
    
    # Get augmentation pipelines
    train_transform = get_augmentation_pipeline('train', image_size)
    val_transform = get_augmentation_pipeline('val', image_size)
    
    # Create datasets
    train_dataset = SARDataset(
        root_dir=root_dir,
        split='train',
        transform=train_transform,
        target_size=(image_size, image_size),
        filter_method=filter_method
    )
    
    val_dataset = SARDataset(
        root_dir=root_dir,
        split='val',
        transform=val_transform,
        target_size=(image_size, image_size),
        filter_method=filter_method
    )
    
    # Only use pin_memory if CUDA is available
    use_pin_memory = torch.cuda.is_available()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader


def process_dataset_pair(sar_path: str, opt_path: Optional[str], output_dir: str,
                        config: DatasetConfig) -> bool:
    """Process a single SAR-Optical pair (or SAR-only if opt_path is None)"""
    try:
        # Create temporary directory
        tmp_dir = os.path.join(output_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Handle SAR-only processing
        if opt_path is None:
            # Check file extension to determine how to read it
            file_ext = Path(sar_path).suffix.lower()
            is_raster_format = file_ext in ['.tif', '.tiff']
            
            if is_raster_format:
                # Use rasterio for GeoTIFF files
                with rasterio.open(sar_path) as sar_ds:
                    sar = sar_ds.read()
                    
                    # Process SAR image
                    if sar.ndim == 3 and sar.shape[0] > 1:
                        sar_img = sar.astype(np.float32)
                    else:
                        sar_img = sar.squeeze(0).astype(np.float32)

                    # Apply filtering
                    if config.filter_method == "lee":
                        sar_img = SARImageProcessor.lee_filter(sar_img)
                    elif config.filter_method == "median":
                        sar_img = SARImageProcessor.median_filter(sar_img)
                    elif config.filter_method == "nlmeans":
                        sar_img = SARImageProcessor.nl_means_filter(sar_img)
                    
                    # Normalize
                    if config.normalization == "robust":
                        sar_norm = SARImageProcessor.robust_normalize(sar_img)
                    elif config.normalization == "zscore":
                        sar_norm = SARImageProcessor.z_score_normalize(sar_img)
                    else:
                        sar_norm = sar_img
                    
                    # Save normalized SAR image as TIF with explicit GTiff driver
                    sar_tmp_path = os.path.join(tmp_dir, Path(sar_path).stem + "_norm.tif")
                    sar_meta = sar_ds.meta.copy()
                    sar_meta.update({
                        'dtype': rasterio.float32,
                        'driver': 'GTiff',  # Explicitly use GTiff driver
                        'compress': 'lzw'
                    })
                    
                    with rasterio.open(sar_tmp_path, "w", **sar_meta) as dst:
                        if sar_norm.ndim == 2:
                            dst.write(sar_norm[np.newaxis, ...].astype(np.float32))
                        else:
                            dst.write(sar_norm.astype(np.float32))
                
                # Create output directory for SAR tiles only
                sar_tiles_dir = os.path.join(output_dir, config.name, "SAR")
                os.makedirs(sar_tiles_dir, exist_ok=True)

                # Tile and save SAR only
                GeoTIFFProcessor.tile_and_save(
                    sar_tmp_path, sar_tiles_dir, 
                    tile_size=config.tile_size, 
                    overlap=config.overlap, 
                    skip_partial=True
                )
            else:
                # Use PIL for JPEG/PNG files
                img = Image.open(sar_path).convert('L')  # Convert to grayscale
                sar_img = np.array(img).astype(np.float32)
                
                # Apply filtering
                if config.filter_method == "lee":
                    sar_img = SARImageProcessor.lee_filter(sar_img)
                elif config.filter_method == "median":
                    sar_img = SARImageProcessor.median_filter(sar_img)
                elif config.filter_method == "nlmeans":
                    sar_img = SARImageProcessor.nl_means_filter(sar_img)
                
                # Normalize
                if config.normalization == "robust":
                    sar_norm = SARImageProcessor.robust_normalize(sar_img)
                elif config.normalization == "zscore":
                    sar_norm = SARImageProcessor.z_score_normalize(sar_img)
                else:
                    sar_norm = sar_img
                
                # Clip to [0, 1] range for image display
                sar_norm = np.clip(sar_norm, 0.0, 1.0)
                
                # Create output directory for SAR tiles only
                sar_tiles_dir = os.path.join(output_dir, config.name, "SAR")
                os.makedirs(sar_tiles_dir, exist_ok=True)
                
                # Tile and save as PNG files directly
                height, width = sar_norm.shape
                step = int(config.tile_size * (1 - config.overlap))
                if step <= 0:
                    step = config.tile_size
                
                tile_count = 0
                skip_partial = True  # Skip partial tiles by default
                for y in range(0, height, step):
                    for x in range(0, width, step):
                        # Check if tile would be partial
                        if (x + config.tile_size > width) or (y + config.tile_size > height):
                            if skip_partial:
                                continue
                            # Handle partial tiles by cropping
                            tile = sar_norm[y:min(y+config.tile_size, height), 
                                           x:min(x+config.tile_size, width)]
                            # Skip if tile is too small
                            if tile.shape[0] < config.tile_size or tile.shape[1] < config.tile_size:
                                continue
                        else:
                            tile = sar_norm[y:y+config.tile_size, x:x+config.tile_size]
                        
                        # Convert to uint8 for PNG (normalized values are in [0, 1])
                        tile_uint8 = (tile * 255).astype(np.uint8)
                        
                        # Save as PNG
                        tile_name = f"{x}_{y}.png"
                        tile_path = os.path.join(sar_tiles_dir, tile_name)
                        Image.fromarray(tile_uint8, mode='L').save(tile_path)
                        tile_count += 1
            
            return True
        
        # Original processing for SAR-Optical pairs
        # Align SAR to Optical
        aligned_sar = os.path.join(tmp_dir, Path(sar_path).stem + "_aligned.tif")
        GeoTIFFProcessor.align_to_reference(sar_path, opt_path, aligned_sar)

        # Process aligned images
        with rasterio.open(aligned_sar) as sar_ds, rasterio.open(opt_path) as opt_ds:
            sar = sar_ds.read()
            opt = opt_ds.read()
            
            # Process SAR image
            if sar.ndim == 3 and sar.shape[0] > 1:
                sar_img = sar.astype(np.float32)
            else:
                sar_img = sar.squeeze(0).astype(np.float32)

            # Apply filtering
            if config.filter_method == "lee":
                sar_img = SARImageProcessor.lee_filter(sar_img)
            elif config.filter_method == "median":
                sar_img = SARImageProcessor.median_filter(sar_img)
            elif config.filter_method == "nlmeans":
                sar_img = SARImageProcessor.nl_means_filter(sar_img)
            
            # Normalize
            if config.normalization == "robust":
                sar_norm = SARImageProcessor.robust_normalize(sar_img)
                opt_norm = SARImageProcessor.robust_normalize(opt.astype(np.float32))
            elif config.normalization == "zscore":
                sar_norm = SARImageProcessor.z_score_normalize(sar_img)
                opt_norm = SARImageProcessor.z_score_normalize(opt.astype(np.float32))
            else:
                sar_norm = sar_img
                opt_norm = opt.astype(np.float32)
            
            # Save normalized images
            sar_tmp_path = os.path.join(tmp_dir, Path(sar_path).stem + "_norm.tif")
            opt_tmp_path = os.path.join(tmp_dir, Path(opt_path).stem + "_norm.tif")

            # Update metadata
            opt_meta = opt_ds.meta.copy()
            opt_meta.update(dtype=rasterio.float32)
            sar_meta = sar_ds.meta.copy()
            sar_meta.update(dtype=rasterio.float32)

            # Save SAR
            with rasterio.open(sar_tmp_path, "w", **sar_meta) as dst:
                if sar_norm.ndim == 2:
                    dst.write(sar_norm[np.newaxis, ...].astype(np.float32))
                else:
                    dst.write(sar_norm.astype(np.float32))
            
            # Save Optical
            with rasterio.open(opt_tmp_path, "w", **opt_meta) as dst:
                dst.write(opt_norm.astype(np.float32))

        # Create output directories
        sar_tiles_dir = os.path.join(output_dir, config.name, "SAR")
        opt_tiles_dir = os.path.join(output_dir, config.name, "Optical")
        os.makedirs(sar_tiles_dir, exist_ok=True)
        os.makedirs(opt_tiles_dir, exist_ok=True)

        # Tile and save
        GeoTIFFProcessor.tile_and_save(
            sar_tmp_path, sar_tiles_dir, 
            tile_size=config.tile_size, 
            overlap=config.overlap, 
            skip_partial=True
        )
        GeoTIFFProcessor.tile_and_save(
            opt_tmp_path, opt_tiles_dir, 
            tile_size=config.tile_size, 
            overlap=config.overlap, 
            skip_partial=True
        )
        
        # Cleanup
        try:
            os.remove(aligned_sar)
        except:
            pass
        
        return True
        
    except Exception as e:
        opt_info = opt_path if opt_path else "SAR-only"
        print(f"Error processing pair {sar_path}, {opt_info}: {e}")
        return False


def resolve_path(path: str) -> Optional[str]:
    """Resolve a path, handling both absolute and relative paths, and path separators"""
    if not path:
        return None
    
    # Normalize path separators
    normalized = path.replace('\\', os.sep).replace('/', os.sep)
    
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
    
    # Try case-insensitive search in common locations
    common_bases = [
        os.getcwd(),
        project_root,
        os.path.join(project_root, "Data", "Raw"),
    ]
    
    path_parts = [p for p in normalized.split(os.sep) if p]
    if len(path_parts) > 0:
        last_part = path_parts[-1]
        for base in common_bases:
            if os.path.exists(base):
                for root, dirs, files in os.walk(base):
                    # Case-insensitive search for directory
                    matching_dirs = [d for d in dirs if d.lower() == last_part.lower()]
                    if matching_dirs:
                        found_path = os.path.join(root, matching_dirs[0])
                        return os.path.abspath(found_path)
    
    return None


def main():
    """Main data processing pipeline"""
    parser = argparse.ArgumentParser(description="SAR Image Colorization Data Pipeline")
    parser.add_argument("--datasets", nargs="+", required=True, 
                       help="Dataset configurations: NAME:SAR_PATH[:OPTICAL_PATH] (optical path is optional)")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="Output directory for processed data")
    parser.add_argument("--tile_size", type=int, default=256, 
                       help="Tile size for processing")
    parser.add_argument("--overlap", type=float, default=0.25, 
                       help="Overlap between tiles")
    parser.add_argument("--filter", type=str, default="lee", 
                       choices=["lee", "median", "nlmeans", "none"],
                       help="SAR filtering method")
    parser.add_argument("--normalization", type=str, default="robust",
                       choices=["robust", "zscore", "none"],
                       help="Normalization method")
    parser.add_argument("--split_ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15],
                       help="Train/Val/Test split ratios")
    
    args = parser.parse_args()
    
    # Validate split ratios
    if abs(sum(args.split_ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process datasets
    processed_datasets = []
    
    for dataset_config in args.datasets:
        parts = dataset_config.split(":")
        if len(parts) < 2 or len(parts) > 3:
            raise ValueError("Dataset config must be NAME:SAR_PATH or NAME:SAR_PATH:OPTICAL_PATH")
        
        if len(parts) == 2:
            name, sar_path = parts
            opt_path = None
        else:
            name, sar_path, opt_path = parts
        
        # Resolve paths
        resolved_sar = resolve_path(sar_path)
        if resolved_sar is None:
            print(f"Warning: Skipping {name} - SAR path not found: {sar_path}")
            print(f"  Attempted to resolve: {sar_path}")
            print(f"  Current working directory: {os.getcwd()}")
            print(f"  Project root: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
            # Suggest common paths
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            common_paths = [
                os.path.join(project_root, "Data", "Raw", "Sentinel-1"),
                os.path.join(project_root, "Data", "Raw", "MSTAR"),
                os.path.join(project_root, "Data", "Raw", "sentinel-1"),
                os.path.join(project_root, "Data", "Raw", "mstar"),
            ]
            existing = [p for p in common_paths if os.path.exists(p)]
            if existing:
                print(f"  Found similar paths: {existing}")
            continue
        
        sar_path = resolved_sar
        
        if opt_path is not None:
            resolved_opt = resolve_path(opt_path)
            if resolved_opt is None:
                print(f"Warning: Skipping {name} - optical path not found: {opt_path}")
                continue
            opt_path = resolved_opt
        
        config = DatasetConfig(
            name=name,
            sar_path=sar_path,
            optical_path=opt_path,
            tile_size=args.tile_size,
            overlap=args.overlap,
            filter_method=args.filter,
            normalization=args.normalization
        )
        
        print(f"Processing dataset: {name}")
        
        # Find SAR image files
        # Recursively gather image files (SEN12-FLOOD stores files in nested scene folders)
        sar_files = sorted([
            f for f in glob(os.path.join(sar_path, "**", "*"), recursive=True)
            if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))
        ])

        if len(sar_files) == 0:
            print(f"No SAR image files found under: {sar_path}")
            continue
        
        # Handle SAR-only datasets (no optical path)
        if opt_path is None:
            print(f"Processing SAR-only dataset: {len(sar_files)} files")
            pairs = [(sar_file, None) for sar_file in sar_files]
        else:
            # Find optical image files
            opt_files = sorted([
                f for f in glob(os.path.join(opt_path, "**", "*"), recursive=True)
                if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))
            ])

            if len(opt_files) == 0:
                print(f"No optical image files found under: {opt_path}")
                continue
            
            # Enhanced pairing: try exact match first, then flexible matching
            pairs = []
            used_opt = set()
            
            for sar_file in sar_files:
                sar_name = Path(sar_file).stem
                matched = False
                
                # Try exact match first
                for opt_file in opt_files:
                    if opt_file in used_opt:
                        continue
                    opt_name = Path(opt_file).stem
                    if sar_name == opt_name:
                        pairs.append((sar_file, opt_file))
                        used_opt.add(opt_file)
                        matched = True
                        break
                
                # Try flexible matching (common patterns for SEN12-FLOOD)
                if not matched:
                    # Remove common prefixes/suffixes
                    sar_base = sar_name.replace('_s1', '').replace('_S1', '').replace('_sar', '').replace('_SAR', '')
                    for opt_file in opt_files:
                        if opt_file in used_opt:
                            continue
                        opt_name = Path(opt_file).stem
                        opt_base = opt_name.replace('_s2', '').replace('_S2', '').replace('_optical', '').replace('_opt', '')
                        
                        # Match if base names are similar (allowing for some variation)
                        if sar_base == opt_base or sar_base in opt_base or opt_base in sar_base:
                            pairs.append((sar_file, opt_file))
                            used_opt.add(opt_file)
                            matched = True
                            break
                    
                    # Try matching by numeric IDs (common in SEN12-FLOOD)
                    sar_id = re.search(r'\d+', sar_name)
                    opt_id = re.search(r'\d+', opt_name)
                    if sar_id and opt_id and sar_id.group() == opt_id.group():
                        pairs.append((sar_file, opt_file))
                        used_opt.add(opt_file)
                        matched = True
                        break
        
        if len(pairs) == 0:
            print(f"No matching pairs found for {name}")
            continue

        print(f"Found {len(pairs)} pairs for {name}")
        
        # Process pairs
        success_count = 0
        for sar_file, opt_file in tqdm(pairs, desc=f"Processing {name}"):
            if process_dataset_pair(sar_file, opt_file, args.output_dir, config):
                success_count += 1
        
        print(f"Successfully processed {success_count}/{len(pairs)} pairs for {name}")
        processed_datasets.append(os.path.join(args.output_dir, name))
    
    # Create train/val/test splits
    print("Creating train/val/test splits...")
    create_dataset_splits(processed_datasets, args.output_dir, args.split_ratios)
    
    # Save metadata
    metadata = {
        "datasets": [Path(d).name for d in processed_datasets],
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "filter": args.filter,
        "normalization": args.normalization,
        "split_ratios": args.split_ratios
    }
    
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Data processing complete. Metadata saved to {metadata_path}")


def create_dataset_splits(dataset_dirs: List[str], output_dir: str, 
                         ratios: List[float]) -> Dict[str, int]:
    """Create train/val/test splits from processed datasets"""
    
    # Collect all pairs
    all_pairs = []
    for dataset_dir in dataset_dirs:
        sar_dir = os.path.join(dataset_dir, "SAR")
        opt_dir = os.path.join(dataset_dir, "Optical")
        
        if not os.path.exists(sar_dir) or not os.path.exists(opt_dir):
            continue
        
        sar_files = sorted(os.listdir(sar_dir))
        opt_files = sorted(os.listdir(opt_dir))
        
        for sar_file in sar_files:
            base_name = os.path.splitext(sar_file)[0]
            for opt_file in opt_files:
                if os.path.splitext(opt_file)[0] == base_name:
                    all_pairs.append((sar_file, opt_file, dataset_dir))
                    break
    
    # Shuffle pairs
    random.shuffle(all_pairs)
    
    # Calculate split sizes
    total_pairs = len(all_pairs)
    train_size = int(ratios[0] * total_pairs)
    val_size = int(ratios[1] * total_pairs)
    
    train_pairs = all_pairs[:train_size]
    val_pairs = all_pairs[train_size:train_size + val_size]
    test_pairs = all_pairs[train_size + val_size:]
    
    # Create splits
    splits = {"train": train_pairs, "val": val_pairs, "test": test_pairs}
    
    for split_name, pairs in splits.items():
        sar_out = os.path.join(output_dir, split_name, "SAR")
        opt_out = os.path.join(output_dir, split_name, "Optical")
        os.makedirs(sar_out, exist_ok=True)
        os.makedirs(opt_out, exist_ok=True)
        
        for sar_file, opt_file, dataset_dir in pairs:
            src_sar = os.path.join(dataset_dir, "SAR", sar_file)
            src_opt = os.path.join(dataset_dir, "Optical", opt_file)
            
            dst_sar = os.path.join(sar_out, sar_file)
            dst_opt = os.path.join(opt_out, opt_file)
            
            shutil.copy2(src_sar, dst_sar)
            shutil.copy2(src_opt, dst_opt)
    
    return {k: len(v) for k, v in splits.items()}


if __name__ == "__main__":
    main()