"""
Process Sentinel-1 dataset for SAR Image Colorization training
"""

import os
import sys
import shutil
import random
from pathlib import Path
from glob import glob

# Add src to path
sys.path.append('src')

def process_sentinel_dataset():
    """Process Sentinel-1 dataset for training"""
    
    # Define paths
    raw_data_dir = "Data/Raw/Sentinel-1/v_2"
    output_dir = "Data/Processed"
    
    # Create output directories
    train_sar_dir = os.path.join(output_dir, "train", "SAR")
    train_opt_dir = os.path.join(output_dir, "train", "Optical")
    val_sar_dir = os.path.join(output_dir, "val", "SAR")
    val_opt_dir = os.path.join(output_dir, "val", "Optical")
    
    for dir_path in [train_sar_dir, train_opt_dir, val_sar_dir, val_opt_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Process each land cover type
    land_cover_types = ['agri', 'barrenland', 'grassland', 'urban']
    
    train_count = 0
    val_count = 0
    
    for land_type in land_cover_types:
        print(f"Processing {land_type}...")
        
        # Get all SAR and optical image pairs
        sar_dir = os.path.join(raw_data_dir, land_type, "s1")
        opt_dir = os.path.join(raw_data_dir, land_type, "s2")
        
        if not os.path.exists(sar_dir) or not os.path.exists(opt_dir):
            print(f"Warning: {land_type} directories not found")
            continue
        
        # Get all SAR files
        sar_files = glob(os.path.join(sar_dir, "*.png"))
        sar_files.sort()
        
        # Get corresponding optical files
        opt_files = []
        for sar_file in sar_files:
            sar_name = os.path.basename(sar_file)
            opt_file = os.path.join(opt_dir, sar_name)
            if os.path.exists(opt_file):
                opt_files.append(opt_file)
            else:
                print(f"Warning: No corresponding optical image for {sar_name}")
        
        # Filter to have matching pairs
        matching_pairs = []
        for sar_file in sar_files:
            sar_name = os.path.basename(sar_file)
            opt_file = os.path.join(opt_dir, sar_name)
            if os.path.exists(opt_file):
                matching_pairs.append((sar_file, opt_file))
        
        print(f"Found {len(matching_pairs)} matching pairs for {land_type}")
        
        # Split into train/val (80/20)
        random.shuffle(matching_pairs)
        split_idx = int(0.8 * len(matching_pairs))
        train_pairs = matching_pairs[:split_idx]
        val_pairs = matching_pairs[split_idx:]
        
        # Copy training data
        for i, (sar_file, opt_file) in enumerate(train_pairs):
            if train_count >= 1000:  # Limit training samples
                break
                
            sar_name = f"train_{train_count:06d}.png"
            opt_name = f"train_{train_count:06d}.png"
            
            shutil.copy2(sar_file, os.path.join(train_sar_dir, sar_name))
            shutil.copy2(opt_file, os.path.join(train_opt_dir, opt_name))
            
            train_count += 1
        
        # Copy validation data
        for i, (sar_file, opt_file) in enumerate(val_pairs):
            if val_count >= 200:  # Limit validation samples
                break
                
            sar_name = f"val_{val_count:06d}.png"
            opt_name = f"val_{val_count:06d}.png"
            
            shutil.copy2(sar_file, os.path.join(val_sar_dir, sar_name))
            shutil.copy2(opt_file, os.path.join(val_opt_dir, opt_name))
            
            val_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {val_count}")
    print(f"Training SAR images: {len(os.listdir(train_sar_dir))}")
    print(f"Training Optical images: {len(os.listdir(train_opt_dir))}")
    print(f"Validation SAR images: {len(os.listdir(val_sar_dir))}")
    print(f"Validation Optical images: {len(os.listdir(val_opt_dir))}")

if __name__ == "__main__":
    process_sentinel_dataset()
