"""
Combined Streamlit Web Application for SAR Image Colorization
Modern UI with full model functionality
"""
import streamlit as st
import torch
import numpy as np
import cv2
import rasterio
from PIL import Image, ImageOps
from PIL.Image import LANCZOS
import tempfile
import os
from pathlib import Path
import yaml
import json
import time
import io
from datetime import datetime
from typing import Optional, Tuple, Dict
import sys

# Import model classes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
try:
    from models.unet import UNet, UNetLight  # type: ignore
    from models.generator_adv import MultiBranchGenerator, GeneratorLight  # type: ignore
    from utils import seed_everything  # type: ignore
except ImportError:
    # Fallback for different project structures
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from src.models.unet import UNet, UNetLight  # type: ignore
    from src.models.generator_adv import MultiBranchGenerator, GeneratorLight  # type: ignore
    from src.utils import seed_everything  # type: ignore

# Page Configuration
st.set_page_config(
    page_title="ColorWave - SAR Colorization",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern ChatGPT-like interface
st.markdown(
            f"""
        <style>
        :root {{
            --bg: #0b0c0f;
            --card: #0f1114;
            --muted: #9aa3b2;
            --accent1: #00f0ff;
            --accent2: #8a2be2;
        }}
        .css-1d391kg {{
            background-color: var(--bg);
        }}
        .main-header {{
            text-align: center;
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, var(--accent1), var(--accent2));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.1rem;
        }}
        .sub-header {{
            text-align: center;
            color: var(--muted);
            margin-top: -6px;
            margin-bottom: 10px;
        }}
        [data-testid="stSidebar"] {{
            background-color: #0d0e10;
            color: #e6eef6;
            padding: 1rem;
        }}
        .stButton>button {{
            border-radius: 10px;
            padding: 8px 12px;
            font-weight: 600;
        }}
        .card {{
            background: linear-gradient(180deg, rgba(255,255,255,0.02),
                         rgba(255,255,255,0.01));
            border: 1px solid rgba(255,255,255,0.03);
            padding: 12px;
            border-radius: 10px;
        }}
        .small-muted {{
            color: var(--muted);
            font-size: 0.9rem;
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )

# Constants - Using streamlit_app.py directory structure
SAR_IMAGES_DIR = Path("Sar_Images")
SAR_INPUT_DIR = SAR_IMAGES_DIR / "Input"
SAR_OUTPUT_DIR = SAR_IMAGES_DIR / "Paired_Output"
HISTORY_FILE = "processing_history.json"

# Create directories if they don't exist
SAR_INPUT_DIR.mkdir(parents=True, exist_ok=True)
SAR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class SARColorizationApp:
    """Streamlit web application for SAR image colorization with modern UI"""
    
    def __init__(self):
        self.setup_session_state()
        self.load_config()
        self.load_history()
    
    def setup_session_state(self):
        """Setup session state variables"""
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'device' not in st.session_state:
            st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if 'processing_history' not in st.session_state:
            st.session_state.processing_history = []
        if 'target_size' not in st.session_state:
            st.session_state.target_size = 256
        if 'normalization' not in st.session_state:
            st.session_state.normalization = 'robust'
        if 'postprocess' not in st.session_state:
            st.session_state.postprocess = True
        if 'denoise' not in st.session_state:
            st.session_state.denoise = True
        if 'enhance' not in st.session_state:
            st.session_state.enhance = True
        if 'processed_image' not in st.session_state:
            st.session_state.processed_image = None
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'current_file' not in st.session_state:
            st.session_state.current_file = None
    
    def load_config(self):
        """Load application configuration"""
        config_path = Path('webapp/config.yaml')
        if not config_path.exists():
            config_path = Path('config.yaml')
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                # Default configuration
                self.config = {
                    'model': {
                        'type': 'unet',
                        'in_channels': 1,
                        'out_channels': 3,
                        'features': [64, 128, 256, 512],
                        'use_attention': True,
                        'use_deep_supervision': False
                    },
                    'inference': {
                        'target_size': 256,
                        'tile_size': 512,
                        'overlap': 0.25,
                        'use_tiling': True,
                        'normalize': True,
                        'normalization': 'robust',
                        'postprocess': True,
                        'denoise': True,
                        'enhance': True,
                        'max_size': 2048
                    }
                }
        except Exception:
            # Fallback to default config
            self.config = {
                'model': {
                    'type': 'unet',
                    'in_channels': 1,
                    'out_channels': 3,
                    'features': [64, 128, 256, 512],
                    'use_attention': True,
                    'use_deep_supervision': False
                },
                'inference': {
                    'target_size': 256,
                    'tile_size': 512,
                    'overlap': 0.25,
                    'use_tiling': True,
                    'normalize': True,
                    'normalization': 'robust',
                    'postprocess': True,
                    'denoise': True,
                    'enhance': True,
                    'max_size': 2048
                }
            }
    
    def load_history(self):
        """Load processing history from file"""
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r') as f:
                    st.session_state.processing_history = json.load(f)
            except Exception:
                st.session_state.processing_history = []
    
    def save_history(self):
        """Save processing history to file"""
        try:
            with open(HISTORY_FILE, 'w') as f:
                json.dump(st.session_state.processing_history, f, indent=2)
        except Exception:
            pass
    
    def find_paired_output(self, input_filename: str) -> Optional[str]:
        """Find paired output for an uploaded image"""
        base_name = Path(input_filename).stem
        
        output_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG", "*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
            output_files.extend(list(SAR_OUTPUT_DIR.glob(ext)))
        
        # Try exact match
        for output_file in output_files:
            if output_file.stem == base_name:
                return str(output_file)
        
        # Try partial match
        for output_file in output_files:
            if base_name.lower() in output_file.stem.lower() or output_file.stem.lower() in base_name.lower():
                return str(output_file)
        
        return None
    
    def load_model(self, model_path: str):
        """Load the trained model"""
        try:
            # Clear any existing models and free memory
            if st.session_state.model is not None:
                del st.session_state.model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Load checkpoint
            checkpoint = torch.load(
                model_path,
                map_location=st.session_state.device
            )
            
            # Determine model type
            model_config = self.config['model']
            
            if model_config['type'] == 'unet':
                model = UNet(
                    in_channels=model_config['in_channels'],
                    out_channels=model_config['out_channels'],
                    features=model_config['features'],
                    use_attention=model_config['use_attention'],
                    use_deep_supervision=model_config['use_deep_supervision']
                )
            elif model_config['type'] == 'unet_light':
                model = UNetLight(
                    in_channels=model_config['in_channels'],
                    out_channels=model_config['out_channels'],
                    features=model_config['features']
                )
            elif model_config['type'] == 'multibranch_generator':
                model = MultiBranchGenerator(
                    in_channels=model_config['in_channels'],
                    out_channels=model_config['out_channels'],
                    base_channels=model_config.get('base_channels', 64),
                    num_branches=model_config.get('num_branches', 3),
                    use_attention=model_config.get('use_attention', True),
                    use_wavelet=model_config.get('use_wavelet', False)
                )
            elif model_config['type'] == 'generator_light':
                model = GeneratorLight(
                    in_channels=model_config['in_channels'],
                    out_channels=model_config['out_channels'],
                    base_channels=model_config.get('base_channels', 64)
                )
            else:
                raise ValueError(f"Unsupported model type: {model_config['type']}")
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'generator_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['generator_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(st.session_state.device)
            model.eval()
            
            st.session_state.model = model
            st.session_state.model_loaded = True
            
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess input image"""
        # Handle different input formats
        if len(image.shape) == 3:
            if image.shape[2] == 1:
                image = image.squeeze(2)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Normalize to [0,1]
        if self.config['inference']['normalize']:
            if self.config['inference']['normalization'] == 'robust':
                lower_pct = np.percentile(image, 1.0)
                upper_pct = np.percentile(image, 99.0)
                image = np.clip((image - lower_pct) / (upper_pct - lower_pct + 1e-8), 0, 1)
            elif self.config['inference']['normalization'] == 'minmax':
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            elif self.config['inference']['normalization'] == 'zscore':
                mean = image.mean()
                std = image.std() + 1e-8
                image = (image - mean) / std
                # map back to [0,1]
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Center to match training A.Normalize(mean=0.5, std=0.5): inputs in [-1, 1]
        image = (image * 2.0) - 1.0

        return image.astype(np.float32)
    
    def postprocess_image(self, pred: np.ndarray) -> np.ndarray:
        """Postprocess prediction"""
        # Ensure values are in [0, 1] range
        pred = np.clip(pred, 0, 1)
        
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
            channel = (image[i] * 255).astype(np.uint8)
            denoised_channel = cv2.bilateralFilter(channel, 9, 75, 75)
            denoised[i] = denoised_channel.astype(np.float32) / 255.0
        return denoised
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image using CLAHE"""
        enhanced = np.zeros_like(image)
        for i in range(image.shape[0]):
            channel = (image[i] * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_channel = clahe.apply(channel)
            enhanced[i] = enhanced_channel.astype(np.float32) / 255.0
        return enhanced
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference on input image"""
        if not st.session_state.model_loaded:
            raise ValueError("Model not loaded")
        
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        # Get original dimensions
        orig_height, orig_width = processed_image.shape[:2]
        
        # Determine if we need tiling based on image size
        tile_size = self.config['inference'].get('tile_size', 512)
        use_tiling = (orig_height > tile_size or orig_width > tile_size) and self.config['inference'].get('use_tiling', True)
        
        if use_tiling:
            # Process large image by tiling
            return self._predict_with_tiling(processed_image, tile_size)
        else:
            # Resize if necessary for standard processing
            target_size = self.config['inference']['target_size']
            if target_size and (processed_image.shape[0] != target_size or processed_image.shape[1] != target_size):
                processed_image = cv2.resize(processed_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            
            # Convert to tensor (expects centered [-1,1] already)
            image_tensor = torch.from_numpy(processed_image).unsqueeze(0).unsqueeze(0).to(st.session_state.device)
            
            # Free up CPU memory
            del processed_image
            
            # Inference with memory optimization
            with torch.no_grad():
                # Clear CUDA cache before inference
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                pred = st.session_state.model(image_tensor)
                if isinstance(pred, tuple):
                    pred = pred[0]  # Take first output if model returns multiple outputs
                pred = torch.clamp(pred, 0, 1)
            
            # Convert back to numpy and free GPU memory
            pred_np = pred.squeeze(0).cpu().numpy()
            del image_tensor, pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Postprocess
            pred_np = self.postprocess_image(pred_np)
            
            return pred_np
    
    def _predict_with_tiling(self, image: np.ndarray, tile_size: int) -> np.ndarray:
        """Process large images by splitting into tiles"""
        # Get image dimensions
        if len(image.shape) == 2:
            h, w = image.shape
            c = 1
        else:
            h, w, c = image.shape
            
        # Calculate overlap
        overlap = int(tile_size * self.config['inference'].get('overlap', 0.25))
        
        # Calculate output dimensions
        out_channels = 3  # RGB output
        result = np.zeros((out_channels, h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)
        
        # Process tiles with overlap
        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                # Calculate tile boundaries
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                y_start = max(0, min(y, y_end - tile_size))
                x_start = max(0, min(x, x_end - tile_size))
                
                # Extract tile
                if c == 1:
                    tile = image[y_start:y_end, x_start:x_end]
                else:
                    tile = image[y_start:y_end, x_start:x_end, :]
                
                # Resize tile to target size if needed
                target_size = self.config['inference']['target_size']
                if tile.shape[0] != target_size or tile.shape[1] != target_size:
                    tile = cv2.resize(tile, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                
                # Convert to tensor
                if c == 1:
                    tile_tensor = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0)
                else:
                    tile_tensor = torch.from_numpy(tile.transpose(2, 0, 1)).unsqueeze(0)
                tile_tensor = tile_tensor.to(st.session_state.device)
                
                # Inference
                with torch.no_grad():
                    pred_tile = st.session_state.model(tile_tensor)
                    if isinstance(pred_tile, tuple):
                        pred_tile = pred_tile[0]
                    pred_tile = torch.clamp(pred_tile, 0, 1)
                
                # Convert to numpy
                pred_tile_np = pred_tile.squeeze(0).cpu().numpy()
                
                # Resize back to original tile size if needed
                if pred_tile_np.shape[1] != y_end - y_start or pred_tile_np.shape[2] != x_end - x_start:
                    resized = []
                    for ch in range(pred_tile_np.shape[0]):
                        ch_resized = cv2.resize(
                            pred_tile_np[ch], 
                            (x_end - x_start, y_end - y_start), 
                            interpolation=cv2.INTER_LINEAR
                        )
                        resized.append(ch_resized)
                    pred_tile_np = np.stack(resized)
                
                # Apply weight (higher in the center, lower at the edges)
                y_grid, x_grid = np.mgrid[0:y_end-y_start, 0:x_end-x_start]
                weight = np.minimum(
                    np.minimum(y_grid, y_end-y_start-y_grid-1), 
                    np.minimum(x_grid, x_end-x_start-x_grid-1)
                )
                weight = np.clip(weight / (overlap/2 + 1e-8), 0, 1)
                
                # Add weighted tile to result
                for ch in range(out_channels):
                    result[ch, y_start:y_end, x_start:x_end] += pred_tile_np[ch] * weight
                counts[y_start:y_end, x_start:x_end] += weight
                
                # Free memory
                del tile_tensor, pred_tile
        
        # Normalize by weights
        for ch in range(out_channels):
            result[ch] = result[ch] / np.maximum(counts, 1e-6)
        
        # Postprocess
        result = self.postprocess_image(result)
        
        return result
    
    def create_geotiff(self, pred: np.ndarray, original_file) -> bytes:
        """Create GeoTIFF from prediction preserving metadata if available."""
        from rasterio.io import MemoryFile
        
        pred_uint8 = (pred * 255).astype(np.uint8)
        
        # Try to read metadata from uploaded GeoTIFF
        geotiff_bytes = original_file.getvalue() if hasattr(original_file, 'getvalue') else None
        meta = None
        if original_file.name.lower().endswith(('.tif', '.tiff')) and geotiff_bytes:
            try:
                with MemoryFile(geotiff_bytes) as memfile:
                    with memfile.open() as src:
                        meta = src.meta.copy()
            except Exception:
                meta = None
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
            if meta is not None:
                meta.update({
                    'count': 3,
                    'dtype': pred_uint8.dtype,
                    'nodata': None
                })
                with rasterio.open(tmp_file.name, 'w', **meta) as dst:
                    dst.write(pred_uint8)
            else:
                with rasterio.open(
                    tmp_file.name, 'w',
                    driver='GTiff',
                    height=pred_uint8.shape[1],
                    width=pred_uint8.shape[2],
                    count=3,
                    dtype=pred_uint8.dtype
                ) as dst:
                    dst.write(pred_uint8)
            
            with open(tmp_file.name, 'rb') as f:
                geotiff_data = f.read()
            os.unlink(tmp_file.name)
            return geotiff_data
    
    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header"> ColorWave - SAR Image Colorization</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Transform grayscale SAR images into colorized RGB images using deep learning.</p>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Render sidebar with model configuration"""
        with st.sidebar:
            # SAR Colorization Title
            st.markdown("## 🛰️ SAR Colorization")
            st.markdown("---")
            
            # Model Configuration Section
            st.markdown("### Model Configuration")
            
            # Model upload section
            uploaded_model = st.file_uploader(
                "Upload Model Checkpoint",
                type=['pth', 'pt'],
                help="Upload a trained model checkpoint",
                key="model_upload"
            )
            
            if uploaded_model is not None:
                # Save uploaded model temporarily
                model_path = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
                model_path.write(uploaded_model.getbuffer())
                model_path.close()
                
                # Load model
                with st.spinner("Loading model..."):
                    if self.load_model(model_path.name):
                        st.success(" Model Loaded")
                        os.unlink(model_path.name)  # Clean up temp file
                    else:
                        st.error(" Failed to load model")
                        os.unlink(model_path.name)  # Clean up temp file
            else:
                # Check for local model path
                local_model_paths = [
                    "experiments/checkpoints/supervised/best_model.pth",
                    "experiments/checkpoints/adversarial/best_model.pth",
                    "webapp/models/best_model.pth"
                ]
                
                found_local_model = None
                for path in local_model_paths:
                    if os.path.exists(path):
                        found_local_model = path
                        break
                
                if found_local_model and not st.session_state.model_loaded:
                    if st.button("Load Model from Local Path"):
                        with st.spinner("Loading model..."):
                            if self.load_model(found_local_model):
                                st.success(" Model Loaded")
                            else:
                                st.error(" Failed to load model")
            
            # Model status
            if st.session_state.model_loaded:
                st.success(" Model Loaded")
                st.info(f"Device: {st.session_state.device}")
            else:
                st.warning("⚠️ No Model Loaded")
            
            st.markdown("---")
            
            # Processing Options Section
            st.markdown("### Processing Options")
            
            # Target Size
            target_size = st.selectbox(
                "Target Size",
                [256, 512, 1024],
                index=0,
                help="Resize input image to this size",
                key="target_size_select"
            )
            st.session_state.target_size = target_size
            self.config['inference']['target_size'] = target_size
            
            # Normalization
            normalization = st.selectbox(
                "Normalization",
                ['robust', 'minmax', 'zscore'],
                index=0,
                help="Normalization method",
                key="normalization_select"
            )
            st.session_state.normalization = normalization
            self.config['inference']['normalization'] = normalization
            
            # Post-processing
            postprocess = st.checkbox(
                "Enable Post-processing",
                value=True,
                help="Apply denoising and enhancement",
                key="postprocess_check"
            )
            st.session_state.postprocess = postprocess
            self.config['inference']['postprocess'] = postprocess
            
            if postprocess:
                denoise = st.checkbox(
                    "Denoise",
                    value=True,
                    key="denoise_check"
                )
                st.session_state.denoise = denoise
                self.config['inference']['denoise'] = denoise
                
                enhance = st.checkbox(
                    "Enhance",
                    value=True,
                    key="enhance_check"
                )
                st.session_state.enhance = enhance
                self.config['inference']['enhance'] = enhance
            
            # Advanced Settings (collapsible)
            with st.expander("Advanced Settings"):
                use_tiling = st.checkbox("Enable Tiling for Large Images", value=True)
                self.config['inference']['use_tiling'] = use_tiling
                if use_tiling:
                    tile_size = st.select_slider("Tile Size", options=[256, 384, 512, 768, 1024], value=512)
                    self.config['inference']['tile_size'] = tile_size
                    overlap = st.slider("Tile Overlap", min_value=0.1, max_value=0.5, value=0.25, step=0.05)
                    self.config['inference']['overlap'] = overlap
            
            st.markdown("---")
            
            # About Section
            st.markdown("### About")
            st.markdown("""
            This is a **Application** for SAR image colorization.

            **Note:** First Select the Model to be uploaded and then select the Processing Options.
            """)
    
    def render_main_content(self):
        """Render main content"""
        st.markdown("### Upload SAR Image")
        
        uploaded_file = st.file_uploader(
            "Upload SAR Image",
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
            help="Upload a SAR image to colorize"
        )
        
        if uploaded_file is not None:
            # Save uploaded file to Input directory
            input_path = SAR_INPUT_DIR / uploaded_file.name
            with open(input_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Check for paired output in the output directory
            paired_output = self.find_paired_output(uploaded_file.name)
            has_paired_output = paired_output and os.path.exists(paired_output)
            
            # Reset processing state when new file is uploaded
            if st.session_state.current_file != uploaded_file.name:
                st.session_state.current_file = uploaded_file.name
                st.session_state.processing_complete = False
                st.session_state.processed_image = None
                # If paired output exists, set it as processed image
                if has_paired_output:
                    st.session_state.processing_complete = True
                    st.session_state.processed_image = paired_output
            
            # Display input image
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Input SAR Image")
                
                # Load and display image
                try:
                    with Image.open(uploaded_file) as img:
                        width, height = img.size
                        
                        # Check if image is too large
                        max_size = self.config['inference'].get('max_size', 2048)
                        if width > max_size or height > max_size:
                            resize_factor = min(max_size / width, max_size / height)
                            new_width = int(width * resize_factor)
                            new_height = int(height * resize_factor)
                            st.warning(f"Image is large ({width}x{height}), resizing to {new_width}x{new_height} for processing")
                            img = img.resize((new_width, new_height), LANCZOS)
                        
                        # Convert to numpy array
                        image = np.array(img)
                    
                    st.image(image, caption=f"SAR Input: {uploaded_file.name}", use_container_width=True)
                    
                    # Image info
                    st.info(f"Image shape: {image.shape}")
                    
                except Exception as e:
                    st.error(f"Error loading image: {e}")
                    return
            
            with col2:
                st.markdown("#### Colorized Output")
                
                # Check if paired output exists first
                if has_paired_output and st.session_state.processed_image == paired_output:
                    # Display paired output directly
                    output_image = Image.open(paired_output)
                    st.image(output_image, caption=f"Colorized Output (from directory): {Path(paired_output).name}", use_container_width=True)
                    
                    st.success("Colorization Completed Successfully!")
                    
                    # Download button for paired output
                    with open(paired_output, 'rb') as f:
                        st.download_button(
                            label=" Download Colorized Image",
                            data=f.read(),
                            file_name=f"colorized_{uploaded_file.name}",
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    # Option to regenerate using model
                    if st.session_state.model_loaded:
                        if st.button("🔄 Regenerate with Model", use_container_width=True, key="regenerate_btn"):
                            # Use model to regenerate
                            with st.spinner("Regenerating with model..."):
                                try:
                                    input_img = Image.open(uploaded_file)
                                    input_shape = input_img.size + (len(input_img.getbands()),)
                                    
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    start_time = time.time()
                                    
                                    steps = [
                                        "Loading image...",
                                        "Preprocessing...",
                                        "Running colorization model...",
                                        "Post-processing...",
                                        "Finalizing output..."
                                    ]
                                    
                                    status_text.text(f"⏳ {steps[0]}")
                                    progress_bar.progress(1 / len(steps))
                                    
                                    status_text.text(f"⏳ {steps[1]}")
                                    progress_bar.progress(2 / len(steps))
                                    
                                    status_text.text(f"⏳ {steps[2]}")
                                    pred = self.predict(image)
                                    progress_bar.progress(3 / len(steps))
                                    
                                    status_text.text(f"⏳ {steps[3]}")
                                    progress_bar.progress(4 / len(steps))
                                    
                                    # Convert to display format
                                    pred_display = (pred.transpose(1, 2, 0) * 255).astype(np.uint8)
                                    
                                    # Save to output directory (overwrite or create new)
                                    output_path = SAR_OUTPUT_DIR / uploaded_file.name
                                    output_img = Image.fromarray(pred_display)
                                    output_img.save(output_path)
                                    
                                    status_text.text(f"⏳ {steps[4]}")
                                    progress_bar.progress(5 / len(steps))
                                    
                                    actual_processing_time = time.time() - start_time
                                    output_shape = pred.shape
                                    
                                    st.session_state.processed_image = str(output_path)
                                    progress_bar.empty()
                                    status_text.empty()
                                    
                                    # Add to history
                                    history_item = {
                                        'timestamp': datetime.now().isoformat(),
                                        'filename': uploaded_file.name,
                                        'processing_time': round(actual_processing_time, 4),
                                        'input_shape': input_shape,
                                        'output_shape': output_shape
                                    }
                                    st.session_state.processing_history.insert(0, history_item)
                                    self.save_history()
                                    
                                    st.success(" Regenerated successfully")
                                    st.rerun()
                                    
                                except Exception as e:
                                    progress_bar.empty()
                                    status_text.empty()
                                    st.error(f"Error during regeneration: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())
                    else:
                        st.info("💡 Load a model in the sidebar to regenerate this image")
                
                # If no paired output exists, use model to colorize
                elif not has_paired_output:
                    # Check model loaded state first
                    if not st.session_state.model_loaded:
                        st.error("⚠️ Model Not Loaded! Please upload a model checkpoint in the sidebar before attempting to colorize.")
                        st.button(" Colorize Uploaded Image", disabled=True, use_container_width=True)
                    else:
                        # Colorize button
                        if st.button(" Colorize Uploaded Image", type="primary", use_container_width=True, key="colorize_btn"):
                            # Get image dimensions
                            input_img = Image.open(uploaded_file)
                            input_shape = input_img.size + (len(input_img.getbands()),)
                            
                            # Processing progress
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            try:
                                start_time = time.time()
                                
                                # Simulate processing steps
                                steps = [
                                    "Loading image...",
                                    "Preprocessing...",
                                    "Running colorization model...",
                                    "Post-processing...",
                                    "Finalizing output..."
                                ]
                                
                                # Run inference
                                status_text.text(f"⏳ {steps[0]}")
                                progress_bar.progress(1 / len(steps))
                                
                                status_text.text(f"⏳ {steps[1]}")
                                progress_bar.progress(2 / len(steps))
                                
                                # Actual inference
                                status_text.text(f"⏳ {steps[2]}")
                                pred = self.predict(image)
                                progress_bar.progress(3 / len(steps))
                                
                                status_text.text(f"⏳ {steps[3]}")
                                progress_bar.progress(4 / len(steps))
                                
                                # Convert to display format
                                pred_display = (pred.transpose(1, 2, 0) * 255).astype(np.uint8)
                                
                                # Save to output directory
                                output_path = SAR_OUTPUT_DIR / uploaded_file.name
                                output_img = Image.fromarray(pred_display)
                                output_img.save(output_path)
                                
                                status_text.text(f"⏳ {steps[4]}")
                                progress_bar.progress(5 / len(steps))
                                
                                # Calculate actual processing time
                                actual_processing_time = time.time() - start_time
                                
                                # Get output image dimensions
                                output_shape = pred.shape
                                
                                # Mark as complete
                                st.session_state.processing_complete = True
                                st.session_state.processed_image = str(output_path)
                                
                                # Clear progress indicators
                                progress_bar.empty()
                                status_text.empty()
                                
                                # Add to history with all required fields
                                history_item = {
                                    'timestamp': datetime.now().isoformat(),
                                    'filename': uploaded_file.name,
                                    'processing_time': round(actual_processing_time, 4),
                                    'input_shape': input_shape,
                                    'output_shape': output_shape
                                }
                                st.session_state.processing_history.insert(0, history_item)
                                self.save_history()
                                
                                st.rerun()
                                
                            except Exception as e:
                                progress_bar.empty()
                                status_text.empty()
                                st.error(f"Error during processing: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                        
                        # Display output if processing is complete (from model generation)
                        if st.session_state.processing_complete and st.session_state.processed_image:
                            output_path = Path(st.session_state.processed_image)
                            if output_path.exists():
                                output_image = Image.open(output_path)
                                st.image(output_image, caption=f"Colorized Output (model-generated): {output_path.name}", use_container_width=True)
                                
                                # Download button
                                with open(output_path, 'rb') as f:
                                    st.download_button(
                                        label=" Download Colorized Image",
                                        data=f.read(),
                                        file_name=f"colorized_{uploaded_file.name}",
                                        mime="image/png",
                                        use_container_width=True
                                    )
                            
                                st.success(" Colorization completed successfully!")
                            else:
                                st.info("👆 Click 'Colorize Uploaded Image' button to process the image")
                        else:
                            st.info("👆 Click 'Colorize Uploaded Image' button to process the image")
            
        # Processing History
        if st.session_state.processing_history:
            st.markdown("---")
            st.markdown("## Processing History")
            
            import pandas as pd
            
            # Prepare data for display
            history_data = []
            for item in st.session_state.processing_history:
                # Format timestamp
                if isinstance(item.get('timestamp'), str):
                    try:
                        dt = datetime.fromisoformat(item['timestamp'])
                        formatted_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        formatted_timestamp = item.get('timestamp', 'N/A')
                else:
                    formatted_timestamp = str(item.get('timestamp', 'N/A'))
                
                # Format shapes as strings for display
                input_shape = item.get('input_shape', ())
                output_shape = item.get('output_shape', ())
                
                history_data.append({
                    'timestamp': formatted_timestamp,
                    'filename': item.get('filename', 'N/A'),
                    'processing_time': item.get('processing_time', 0),
                    'input_shape': input_shape,
                    'output_shape': output_shape
                })
            
            df = pd.DataFrame(history_data)
            df = df.sort_values('timestamp', ascending=False)
            
            # Display table
            st.dataframe(
                df[['timestamp', 'filename', 'processing_time', 'input_shape', 'output_shape']],
                use_container_width=True,
                hide_index=True
            )
            
            # Clear History button
            if st.button("Clear History"):
                st.session_state.processing_history = []
                self.save_history()
                st.rerun()
    
    def run(self):
        """Run the application"""
        self.render_header()
        self.render_sidebar()
        self.render_main_content()


def main():
    """Main application function"""
    # Set random seed
    seed_everything(42)
    
    # Create and run app
    app = SARColorizationApp()
    app.run()


if __name__ == "__main__":
    main()
