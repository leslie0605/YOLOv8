import argparse
from pathlib import Path
import shutil
import os
import glob


def main(opt):
    """Simple visualization function that just copies detection images to the output directory"""
    # Input directory
    results_dir = Path(opt.results_dir)
    
    # Create output directory
    if opt.output_dir:
        output_dir = Path(opt.output_dir)
    else:
        output_dir = results_dir / 'viz'
    
    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Visualizations will be saved to: {output_dir}")
    
    # Look for detection images in the results directory
    image_patterns = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
    found_images = []
    
    # Try different subdirectories where detection images might be found
    search_dirs = [
        results_dir,
        results_dir / 'viz',
        results_dir / 'pred',
        results_dir / 'predictions',
    ]
    
    # Also look for specific YOLOv8 output directories
    for subdir in results_dir.glob('*'):
        if subdir.is_dir() and 'detect' in subdir.name.lower():
            search_dirs.append(subdir)
    
    # Search for images in all possible locations
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
            
        for pattern in image_patterns:
            for img_path in search_dir.glob(pattern):
                found_images.append(img_path)
    
    # Copy images to the output directory
    if found_images:
        print(f"Found {len(found_images)} detection images")
        for img_path in found_images:
            dest_path = output_dir / img_path.name
            shutil.copy2(img_path, dest_path)
            print(f"Copied: {img_path.name}")
    else:
        print("No detection images found in the results directory")
        
        # If no images found, try to look for them in more places
        print("Searching for detection images in the project directory...")
        
        # Check for YOLOv8 default output directory
        yolo_dirs = list(Path('.').glob('runs/detect/*'))
        for yolo_dir in yolo_dirs:
            if yolo_dir.is_dir():
                for pattern in image_patterns:
                    for img_path in yolo_dir.glob(pattern):
                        dest_path = output_dir / img_path.name
                        shutil.copy2(img_path, dest_path)
                        print(f"Copied from YOLOv8 output: {img_path.name}")
    
    # Count how many images were copied
    copied_images = list(output_dir.glob('*.jpg')) + list(output_dir.glob('*.png'))
    if copied_images:
        print(f"Successfully copied {len(copied_images)} images to {output_dir}")
    else:
        print("No images were copied. Please check that detection images exist.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy detection images to visualization directory")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing detection results")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save visualizations")
    
    main(parser.parse_args()) 