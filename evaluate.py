import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO
from models.attention import register_attention  # noqa


def find_data_yaml():
    """Find the appropriate data.yaml file"""
    # Check common locations for data.yaml
    possible_paths = [
        Path("data/data.yaml"),
        Path("data.yaml"),
        Path("../data/data.yaml"),
        Path("configs/data.yaml"),
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"Found data configuration at: {path}")
            return path
    
    # Default fallback
    print("Warning: Could not find data.yaml, using default path 'data/data.yaml'")
    return Path("data/data.yaml")


def main(opt):
    # Register CBAM attention module
    register_attention()
    
    # Create output directory structure
    output_dir = Path(opt.project) / opt.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trained model
    model = YOLO(opt.weights)
    print(f"Loaded model from {opt.weights}")
    
    # Find data configuration
    data_yaml = find_data_yaml()
    
    # Run evaluation
    results = model.val(
        data=data_yaml,
        split="test",  # Evaluate on test set
        imgsz=640,
        batch=opt.batch,
        device=opt.device or None,
        verbose=opt.verbose,
        save_json=True,
        save_conf=True,
        save=True,  # Save prediction results
        project=opt.project,
        name=opt.name,
        exist_ok=True,
    )
    
    # Visualize prediction results on some test images
    if opt.visualize > 0:
        print(f"\nVisualizing predictions on {opt.visualize} test images...")
        
        # Create visualization subdirectory
        viz_dir = output_dir / "viz"
        viz_dir.mkdir(exist_ok=True)
        
        # Find test images directory
        test_images_dir = data_yaml.parent / "test" / "images"
        if not test_images_dir.exists():
            # Try alternative path based on data.yaml structure
            with open(data_yaml, 'r') as f:
                for line in f:
                    if line.strip().startswith('test:'):
                        test_path = line.split(':')[1].strip().split('#')[0].strip()
                        possible_test_dir = data_yaml.parent / test_path
                        if possible_test_dir.exists():
                            test_images_dir = possible_test_dir
                            break
        
        if not test_images_dir.exists():
            print(f"Warning: Could not find test images directory. Tried {test_images_dir}")
            test_images_dir = Path("data/test/images")  # Fallback
            
        print(f"Using test images from: {test_images_dir}")
        
        test_results = model(
            source=test_images_dir,
            conf=0.25,
            save=True,
            project=str(output_dir),  # Save in the same output directory
            name="viz",               # Put visualizations in a viz subdirectory
            exist_ok=True,
            max_det=100,
            save_conf=True,
            save_crop=opt.save_crops,
            stream=True,
            verbose=False,  # Reduce output noise
        )
        
        # Visualize only specified number of images
        for i, result in enumerate(test_results):
            if i >= opt.visualize:
                break
                
            # Save prediction visualization
            result.save()
            
        print(f"Visualization results saved to {output_dir}/viz")
    
    # Print evaluation metrics
    metrics = results.box  # Get object detection metrics
    
    # Print main metrics
    print("\nEvaluation Results:")
    print(f"Overall mAP@0.5:0.95: {metrics.map:.4f}")
    print(f"Overall mAP@0.5: {metrics.map50:.4f}")
    print(f"Overall mAP@0.75: {metrics.map75:.4f}")
    
    # Fix the precision and recall metrics - they might be arrays
    if hasattr(metrics, 'p') and isinstance(metrics.p, np.ndarray):
        print(f"Precision: {float(metrics.p.mean()):.4f}")
    elif hasattr(metrics, 'p'):
        print(f"Precision: {metrics.p:.4f}")
        
    if hasattr(metrics, 'r') and isinstance(metrics.r, np.ndarray):
        print(f"Recall: {float(metrics.r.mean()):.4f}")
    elif hasattr(metrics, 'r'):
        print(f"Recall: {metrics.r:.4f}")
    
    
    # Save text summary
    with open(output_dir / "metrics_summary.txt", "w") as f:
        f.write(f"Model: {opt.weights}\n")
        f.write(f"Data configuration: {data_yaml}\n")
        f.write(f"mAP@0.5:0.95: {metrics.map:.4f}\n")
        f.write(f"mAP@0.5: {metrics.map50:.4f}\n")
        f.write(f"mAP@0.75: {metrics.map75:.4f}\n")
        
        if hasattr(metrics, 'p'):
            p_value = float(metrics.p.mean()) if isinstance(metrics.p, np.ndarray) else metrics.p
            f.write(f"Precision: {p_value:.4f}\n")
            
        if hasattr(metrics, 'r'):
            r_value = float(metrics.r.mean()) if isinstance(metrics.r, np.ndarray) else metrics.r
            f.write(f"Recall: {r_value:.4f}\n")
    
    print(f"Evaluation complete, results saved to {output_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Model weights path")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Device, e.g., '0' or 'cpu'")
    parser.add_argument("--visualize", type=int, default=10, help="Number of images to visualize")
    parser.add_argument("--save-crops", action="store_true", help="Save crops of detected objects")
    parser.add_argument("--verbose", action="store_true", help="Show detailed information")
    parser.add_argument("--project", type=str, default="outputs", help="Project directory for saving results")
    parser.add_argument("--name", type=str, default="eval", help="Run name for saving results")
    parser.add_argument("--plot-per-class", action="store_true", help="Plot per-class performance")
    
    main(parser.parse_args()) 