"""
Production-Ready Inference Script for SAR Image Colorization
Batch inference with GeoTIFF support and comprehensive output options
"""

import os
import yaml
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union
import time
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm
import cv2
from PIL import Image

# Suppress NotGeoreferencedWarning for images without geospatial metadata
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

# Support running as a package and as a script
try:
    from .models.unet import UNet, UNetLight
    from .models.generator_adv import MultiBranchGenerator, GeneratorLight
    from .utils import seed_everything, load_checkpoint
except ImportError:
    from models.unet import UNet, UNetLight
    from models.generator_adv import MultiBranchGenerator, GeneratorLight
    from utils import seed_everything, load_checkpoint


class SARInferenceEngine:
    """Production-ready inference engine for SAR image colorization"""
    
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
        
    def setup_logging(self):
        """Setup logging"""
        log_dir = Path(self.config['experiment']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'inference.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Setup output directories"""
        # Prefer inference.output_dir; fallback to experiment.output_dir; else default path
        default_dir = self.config.get('experiment', {}).get('output_dir', 'experiments/outputs/inference')
        self.output_dir = Path(self.config.get('inference', {}).get('output_dir', default_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'png').mkdir(exist_ok=True)
        (self.output_dir / 'geotiff').mkdir(exist_ok=True)
        (self.output_dir / 'comparison').mkdir(exist_ok=True)
    
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
        
        self.logger.info(f"Model loaded successfully on {self.device}")
        
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, dict]:
        """Preprocess input image"""
        file_ext = Path(image_path).suffix.lower()
        is_raster_format = file_ext in ['.tif', '.tiff']
        
        if is_raster_format:
            # Use rasterio for GeoTIFF files
            try:
                with rasterio.open(image_path) as src:
                    image = src.read().astype(np.float32)
                    metadata = src.meta.copy()
                    
                    # Handle different channel configurations
                    if image.ndim == 3 and image.shape[0] == 1:
                        image = image.squeeze(0)
                    elif image.ndim == 3 and image.shape[0] > 1:
                        image = image[0]  # Take first channel
            except Exception as e:
                # Fallback to PIL if rasterio fails
                self.logger.warning(f"Failed to open with rasterio, trying PIL: {e}")
                img = Image.open(image_path).convert('L')
                image = np.array(img).astype(np.float32)
                metadata = {
                    'width': image.shape[1],
                    'height': image.shape[0],
                    'count': 1,
                    'dtype': 'float32',
                    'crs': None,
                    'transform': None
                }
        else:
            # Use PIL for PNG/JPEG files
            img = Image.open(image_path).convert('L')
            image = np.array(img).astype(np.float32)
            metadata = {
                'width': image.shape[1],
                'height': image.shape[0],
                'count': 1,
                'dtype': 'float32',
                'crs': None,
                'transform': None
            }
        
        # Normalize
        if self.config['inference']['normalize']:
            image = self.normalize_image(image)
        
        return image, metadata
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image and center to match training stats."""
        if self.config['inference']['normalization'] == 'robust':
            # Robust normalization using percentiles
            lower_pct = np.percentile(image, 1.0)
            upper_pct = np.percentile(image, 99.0)
            denom = upper_pct - lower_pct
            if denom <= 1e-8:
                image = np.zeros_like(image, dtype=np.float32)
            else:
                image = np.clip((image - lower_pct) / denom, 0, 1)
        elif self.config['inference']['normalization'] == 'minmax':
            # Min-max normalization
            min_v = image.min()
            max_v = image.max()
            denom = max_v - min_v
            if denom <= 1e-8:
                image = np.zeros_like(image, dtype=np.float32)
            else:
                image = (image - min_v) / denom
        elif self.config['inference']['normalization'] == 'zscore':
            # Z-score normalization
            std = image.std()
            if std <= 1e-8:
                image = np.zeros_like(image, dtype=np.float32)
            else:
                image = (image - image.mean()) / std
            # Scale to [0,1] with guard
            min_v = image.min()
            max_v = image.max()
            denom = max_v - min_v
            if denom <= 1e-8:
                image = np.zeros_like(image, dtype=np.float32)
            else:
                image = (image - min_v) / denom
        
        # Ensure finite and within [0,1]
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        # Center to [-1, 1] to match training A.Normalize(mean=0.5, std=0.5)
        image = image * 2.0 - 1.0
        return image
    
    def tile_inference(self, image: np.ndarray, tile_size: int = 256, 
                      overlap: float = 0.25) -> np.ndarray:
        """Perform tiled inference on large images"""
        h, w = image.shape
        step = int(tile_size * (1 - overlap))
        
        # Calculate output dimensions
        out_h = ((h - tile_size) // step + 1) * step + tile_size
        out_w = ((w - tile_size) // step + 1) * step + tile_size
        
        # Pad image if necessary
        if h < out_h or w < out_w:
            pad_h = max(0, out_h - h)
            pad_w = max(0, out_w - w)
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        # Initialize output
        output = np.zeros((3, h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)
        
        # Process tiles
        for y in range(0, h, step):
            for x in range(0, w, step):
                # Extract tile
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                tile = image[y:y_end, x:x_end]
                
                # Pad tile if necessary
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    pad_y = tile_size - tile.shape[0]
                    pad_x = tile_size - tile.shape[1]
                    tile = np.pad(tile, ((0, pad_y), (0, pad_x)), mode='reflect')
                
                # Convert to tensor (tile already centered if source was)
                tile_tensor = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(self.device)
                
                # Inference
                with torch.no_grad():
                    pred_tile = self.model(tile_tensor)
                    pred_tile = torch.clamp(pred_tile, 0, 1)
                
                # Convert back to numpy
                pred_np = pred_tile.squeeze(0).cpu().numpy()
                
                # Crop to original tile size
                pred_np = pred_np[:, :y_end-y, :x_end-x]
                
                # Add to output
                output[:, y:y_end, x:x_end] += pred_np
                count[y:y_end, x:x_end] += 1
        
        # Average overlapping regions
        output = output / np.maximum(count, 1)
        
        return output
    
    def single_inference(self, image: np.ndarray) -> np.ndarray:
        """Perform inference on single image"""
        # Resize if necessary
        target_size = self.config['inference']['target_size']
        if target_size and (image.shape[0] != target_size or image.shape[1] != target_size):
            image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor (expects centered [-1,1])
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            pred = self.model(image_tensor)
            pred = torch.clamp(pred, 0, 1)
        
        # Convert back to numpy
        pred_np = pred.squeeze(0).cpu().numpy()
        
        return pred_np
    
    def postprocess_image(self, pred: np.ndarray, metadata: dict) -> np.ndarray:
        """Postprocess prediction"""
        # Apply post-processing
        if self.config['inference']['postprocess']:
            # Denoise
            if self.config['inference']['denoise']:
                pred = self.denoise_image(pred)
            
            # Enhance
            if self.config['inference']['enhance']:
                pred = self.enhance_image(pred)
        
        return pred
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Denoise image using bilateral filter"""
        denoised = np.zeros_like(image)
        for i in range(image.shape[0]):
            channel_f = np.nan_to_num(image[i], nan=0.0, posinf=1.0, neginf=0.0)
            channel_f = np.clip(channel_f, 0.0, 1.0)
            channel = (channel_f * 255).astype(np.uint8)
            denoised_channel = cv2.bilateralFilter(channel, 9, 75, 75)
            denoised[i] = denoised_channel.astype(np.float32) / 255.0
        return denoised
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image using CLAHE"""
        enhanced = np.zeros_like(image)
        for i in range(image.shape[0]):
            channel_f = np.nan_to_num(image[i], nan=0.0, posinf=1.0, neginf=0.0)
            channel_f = np.clip(channel_f, 0.0, 1.0)
            channel = (channel_f * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_channel = clahe.apply(channel)
            enhanced[i] = enhanced_channel.astype(np.float32) / 255.0
        return enhanced
    
    def save_png(self, pred: np.ndarray, output_path: str):
        """Save prediction as PNG"""
        # Convert to PIL format
        pred_sanitized = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
        pred_sanitized = np.clip(pred_sanitized, 0.0, 1.0)
        pred_pil = (pred_sanitized.transpose(1, 2, 0) * 255).astype(np.uint8)
        pred_pil = Image.fromarray(pred_pil)
        
        # Save
        pred_pil.save(output_path, 'PNG')
    
    def save_geotiff(self, pred: np.ndarray, metadata: dict, output_path: str):
        """Save prediction as GeoTIFF with metadata"""
        # Update metadata
        output_metadata = metadata.copy()
        output_metadata.update({
            'count': 3,
            'dtype': 'uint8',
            'driver': 'GTiff',
            'compress': 'lzw',
            'nodata': None
        })
        
        # Set default transform if missing (for PNG/JPEG files without geospatial data)
        if 'transform' not in output_metadata or output_metadata['transform'] is None:
            # Use identity transform (pixel coordinates)
            output_metadata['transform'] = rasterio.Affine.identity()
            output_metadata['crs'] = None
        
        # Ensure width and height are set
        if 'width' not in output_metadata:
            output_metadata['width'] = pred.shape[2]
        if 'height' not in output_metadata:
            output_metadata['height'] = pred.shape[1]
        
        # Convert to uint8
        pred_sanitized = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
        pred_sanitized = np.clip(pred_sanitized, 0.0, 1.0)
        pred_uint8 = (pred_sanitized * 255).astype(np.uint8)
        
        # Save with warning suppression
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
            with rasterio.open(output_path, 'w', **output_metadata) as dst:
                dst.write(pred_uint8)
    
    def create_comparison(self, sar: np.ndarray, pred: np.ndarray, output_path: str):
        """Create comparison visualization"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # SAR image
        axes[0].imshow(sar, cmap='gray')
        axes[0].set_title('SAR Input')
        axes[0].axis('off')
        
        # Prediction
        axes[1].imshow(pred.transpose(1, 2, 0))
        axes[1].set_title('Colorized Output')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def process_single_image(self, input_path: str, output_prefix: str) -> dict:
        """Process a single image"""
        start_time = time.time()
        
        # Load and preprocess
        sar, metadata = self.preprocess_image(input_path)
        
        # Inference
        if self.config['inference']['tile_size'] and \
           (sar.shape[0] > self.config['inference']['tile_size'] or 
            sar.shape[1] > self.config['inference']['tile_size']):
            pred = self.tile_inference(
                sar, 
                self.config['inference']['tile_size'],
                self.config['inference']['overlap']
            )
        else:
            pred = self.single_inference(sar)
        
        # Postprocess
        pred = self.postprocess_image(pred, metadata)
        
        # Save outputs
        png_path = self.output_dir / 'png' / f'{output_prefix}.png'
        geotiff_path = self.output_dir / 'geotiff' / f'{output_prefix}.tif'
        comparison_path = self.output_dir / 'comparison' / f'{output_prefix}_comparison.png'
        
        self.save_png(pred, str(png_path))
        self.save_geotiff(pred, metadata, str(geotiff_path))
        self.create_comparison(sar, pred, str(comparison_path))
        
        processing_time = time.time() - start_time
        
        return {
            'input_path': input_path,
            'output_prefix': output_prefix,
            'processing_time': processing_time,
            'output_shape': pred.shape,
            'png_path': str(png_path),
            'geotiff_path': str(geotiff_path),
            'comparison_path': str(comparison_path)
        }
    
    def process_batch(self, input_paths: List[str]) -> List[dict]:
        """Process a batch of images"""
        self.logger.info(f"Processing {len(input_paths)} images...")
        
        results = []
        pbar = tqdm(input_paths, desc="Inference")
        
        for i, input_path in enumerate(pbar):
            try:
                # Generate output prefix
                output_prefix = Path(input_path).stem
                
                # Process image
                result = self.process_single_image(input_path, output_prefix)
                results.append(result)
                
                # Update progress
                pbar.set_postfix({
                    'Time': f"{result['processing_time']:.2f}s",
                    'Shape': f"{result['output_shape'][1]}x{result['output_shape'][2]}"
                })
                
            except Exception as e:
                self.logger.error(f"Error processing {input_path}: {e}")
                results.append({
                    'input_path': input_path,
                    'error': str(e)
                })
        
        return results
    
    def _resolve_input_paths(self, input_path: str) -> List[str]:
        """Resolve input path(s) - handle both files and directories"""
        # Normalize path
        normalized = input_path.replace('\\', os.sep).replace('/', os.sep)
        
        # Try to resolve relative paths
        resolved_path = None
        if os.path.exists(normalized):
            resolved_path = os.path.abspath(normalized)
        else:
            # Try relative to current directory
            rel_path = os.path.join(os.getcwd(), normalized.lstrip(os.sep))
            if os.path.exists(rel_path):
                resolved_path = os.path.abspath(rel_path)
            else:
                # Try relative to project root
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                rel_to_project = os.path.join(project_root, normalized.lstrip(os.sep))
                if os.path.exists(rel_to_project):
                    resolved_path = os.path.abspath(rel_to_project)
        
        if resolved_path is None:
            raise FileNotFoundError(f"Input path not found: {input_path}")
        
        # Check if it's a directory or file
        if os.path.isdir(resolved_path):
            # Find all image files recursively
            image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.TIF', '.TIFF', '.PNG', '.JPG', '.JPEG']
            image_files = []
            for root, dirs, files in os.walk(resolved_path):
                for file in files:
                    if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))
            
            if not image_files:
                raise ValueError(f"No image files found in directory: {resolved_path}")
            
            self.logger.info(f"Found {len(image_files)} images in directory: {resolved_path}")
            return sorted(image_files)
        elif os.path.isfile(resolved_path):
            # Single file
            return [resolved_path]
        else:
            raise ValueError(f"Input path is neither a file nor a directory: {resolved_path}")
    
    def run_inference(self, input_paths: Union[str, List[str]]):
        """Run inference on input paths"""
        if isinstance(input_paths, str):
            # Resolve input paths (handles directories and files)
            input_paths = self._resolve_input_paths(input_paths)
        
        # Process batch
        results = self.process_batch(input_paths)
        
        # Save results summary
        self.save_results_summary(results)
        
        self.logger.info(f"Inference completed! Results saved to {self.output_dir}")
    
    def save_results_summary(self, results: List[dict]):
        """Save results summary"""
        import json
        
        # Calculate statistics
        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]
        
        if successful_results:
            avg_time = np.mean([r['processing_time'] for r in successful_results])
            total_time = sum([r['processing_time'] for r in successful_results])
        else:
            avg_time = 0
            total_time = 0
        
        summary = {
            'total_images': len(results),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'average_processing_time': avg_time,
            'total_processing_time': total_time,
            'results': results
        }
        
        # Save summary
        with open(self.output_dir / 'inference_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Log summary
        self.logger.info(f"Processed {len(successful_results)}/{len(results)} images successfully")
        self.logger.info(f"Average processing time: {avg_time:.2f}s")
        self.logger.info(f"Total processing time: {total_time:.2f}s")


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="SAR Image Colorization Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to input image(s)")
    parser.add_argument("--device", type=str, default='auto', help="Device to use (auto/cpu/cuda)")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    seed_everything(config['experiment']['seed'])
    
    # Create inference engine
    engine = SARInferenceEngine(config, args.model, args.device)
    
    # Run inference
    engine.run_inference(args.input)


if __name__ == "__main__":
    main()