#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path
import time
import shutil
import yaml


def run_command(cmd, description=None):
    """Run a command and print its output in real-time"""
    if description:
        print(f"\n{'='*80}\n{description}\n{'='*80}")
    
    print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode


def setup_experiment_folders(experiment_name):
    """Create a folder structure for the experiment results"""
    base_dir = Path(f"outputs/experiments/{experiment_name}")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create folders for each model type
    for model_type in ["yolo_pretrained", "yolo_scratch", "yolo_cbam_scratch"]:
        model_dir = base_dir / model_type
        model_dir.mkdir(exist_ok=True)
    
    return base_dir


def train_model(model_type, epochs, batch_size, device, exp_dir):
    """Train a specific model configuration"""
    print(f"\n\n{'#'*100}\nTraining {model_type}\n{'#'*100}")
    
    # Create output directory for this model
    model_dir = exp_dir / model_type
    model_dir.mkdir(exist_ok=True)
    
    # Detect the data.yaml location
    data_yaml = find_data_yaml()
    
    if model_type == "yolo_pretrained":
        # Standard YOLOv8 pretrained
        result = run_command(
            ["yolo", "train", "task=detect", 
             f"model=yolov8n.pt", 
             f"data={data_yaml}", 
             f"epochs={epochs}", 
             f"batch={batch_size}",
             f"device={device}",
             f"project={model_dir}",
             "name=train",
             "exist_ok=True"],
            f"Training standard YOLOv8 with pretrained weights"
        )
    
    elif model_type == "yolo_scratch":
        # Standard YOLOv8 from scratch
        result = run_command(
            ["yolo", "train", "task=detect", 
             f"model=yolov8n.yaml", 
             f"data={data_yaml}", 
             f"epochs={epochs}", 
             f"batch={batch_size}",
             f"device={device}",
             f"project={model_dir}",
             "name=train",
             "exist_ok=True"],
            f"Training standard YOLOv8 from scratch"
        )
    
    elif model_type == "yolo_cbam_scratch":
        # YOLOv8 with CBAM attention, from scratch
        result = run_command(
            ["python", "train.py", 
             f"--epochs={epochs}", 
             f"--batch={batch_size}",
             f"--device={device}",
             f"--project={model_dir}",
             "--name=train"],
            f"Training YOLOv8 with CBAM attention from scratch"
        )
    
    return result == 0  # Return True if command succeeded


def evaluate_model(model_type, weights_path, batch_size, device, exp_dir):
    """Evaluate a trained model on the test set"""
    print(f"\n\n{'#'*100}\nEvaluating {model_type}\n{'#'*100}")
    
    # Create output directory for evaluation results
    model_dir = exp_dir / model_type
    eval_dir = model_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    
    # Detect the data.yaml location
    data_yaml = find_data_yaml()
    
    if model_type in ["yolo_pretrained", "yolo_scratch"]:
        # Standard YOLOv8 evaluation
        result = run_command(
            ["yolo", "val", "task=detect", 
             f"model={weights_path}", 
             f"data={data_yaml}", 
             "split=test",
             f"batch={batch_size}",
             f"device={device}",
             "save_json=True",
             "save_conf=True",
             "save=True",
             f"project={eval_dir}",
             "name=eval_results",  # Use a different name to avoid conflict with built-in eval
             "exist_ok=True"],
            f"Evaluating {model_type}"
        )
    
    else:
        # YOLOv8 with CBAM attention evaluation
        result = run_command(
            ["python", "evaluate.py", 
             f"--weights={weights_path}", 
             f"--batch={batch_size}",
             f"--device={device}",
             "--verbose",
             "--save-crops",
             "--plot-per-class",
             f"--project={eval_dir}",
             "--name=eval_results"],
            f"Evaluating {model_type}"
        )
    
    return result == 0  # Return True if command succeeded


def visualize_results(model_type, eval_dir, exp_dir):
    """Visualize evaluation results using the visualization script"""
    print(f"\n\n{'#'*100}\nVisualizing results for {model_type}\n{'#'*100}")
    
    # Create output directory for visualizations
    model_dir = exp_dir / model_type
    viz_dir = model_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    
    result = run_command(
        ["python", "visualize.py", 
         f"--results-dir={eval_dir}/eval_results", 
         f"--output-dir={viz_dir}"],
        f"Visualizing results for {model_type}"
    )
    
    return result == 0  # Return True if command succeeded


def extract_metrics(eval_dir, model_type):
    """Extract metrics from evaluation results"""
    # Try to find JSON results
    json_files = list(Path(eval_dir).glob("*.json"))
    
    if json_files:
        try:
            with open(json_files[0], 'r') as f:
                data = yaml.safe_load(f)
                
                # Extract metrics
                metrics = {}
                if "metrics/mAP50-95" in data:
                    metrics["map"] = data["metrics/mAP50-95"]
                if "metrics/mAP50" in data:
                    metrics["map50"] = data["metrics/mAP50"]
                if "metrics/mAP75" in data:
                    metrics["map75"] = data.get("metrics/mAP75", 0)
                if "metrics/precision" in data:
                    metrics["precision"] = data["metrics/precision"]
                if "metrics/recall" in data:
                    metrics["recall"] = data["metrics/recall"]
                
                return metrics
        except Exception as e:
            print(f"Error extracting metrics for {model_type}: {e}")
    
    return None


def find_data_yaml():
    """Find the appropriate data.yaml file based on the project structure"""
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
            return str(path)
    
    # Default fallback
    print("Warning: Could not find data.yaml, using default path 'data/data.yaml'")
    return "data/data.yaml"


def main(opt):
    start_time = time.time()
    
    # Setup experiment directory
    exp_dir = setup_experiment_folders(opt.experiment_name)
    print(f"Experiment results will be saved to: {exp_dir}")
    
    # Process each model if selected
    if opt.all or opt.yolo_pretrained:
        if train_model("yolo_pretrained", opt.epochs, opt.batch, opt.device, exp_dir):
            weights_path = str(Path(exp_dir / "yolo_pretrained/train/weights/best.pt"))
            eval_dir = str(Path(exp_dir / "yolo_pretrained/eval"))
            
            # Ensure the weights path exists before proceeding
            weights_file = Path(weights_path)
            if not weights_file.exists():
                print(f"Warning: Weights file not found at {weights_path}")
                # Try to find it in alternative locations
                alt_paths = [
                    Path(exp_dir / "yolo_pretrained/train/train/weights/best.pt"),
                    Path("runs/detect/train/weights/best.pt")
                ]
                
                for alt_path in alt_paths:
                    if alt_path.exists():
                        weights_path = str(alt_path)
                        print(f"Using alternative weights file: {weights_path}")
                        break
            
            if evaluate_model("yolo_pretrained", weights_path, opt.batch, opt.device, exp_dir):
                visualize_results("yolo_pretrained", eval_dir, exp_dir)
    
    if opt.all or opt.yolo_scratch:
        if train_model("yolo_scratch", opt.epochs, opt.batch, opt.device, exp_dir):
            weights_path = str(Path(exp_dir / "yolo_scratch/train/weights/best.pt"))
            eval_dir = str(Path(exp_dir / "yolo_scratch/eval"))
            
            # Ensure the weights path exists before proceeding
            weights_file = Path(weights_path)
            if not weights_file.exists():
                print(f"Warning: Weights file not found at {weights_path}")
                # Try to find it in alternative locations
                alt_paths = [
                    Path(exp_dir / "yolo_scratch/train/train/weights/best.pt"),
                    Path("runs/detect/train/weights/best.pt")
                ]
                
                for alt_path in alt_paths:
                    if alt_path.exists():
                        weights_path = str(alt_path)
                        print(f"Using alternative weights file: {weights_path}")
                        break
            
            if evaluate_model("yolo_scratch", weights_path, opt.batch, opt.device, exp_dir):
                visualize_results("yolo_scratch", eval_dir, exp_dir)
    
    if opt.all or opt.yolo_cbam_scratch:
        if train_model("yolo_cbam_scratch", opt.epochs, opt.batch, opt.device, exp_dir):
            weights_path = str(Path(exp_dir / "yolo_cbam_scratch/train/weights/best.pt"))
            eval_dir = str(Path(exp_dir / "yolo_cbam_scratch/eval"))
            
            # Ensure the weights path exists
            weights_file = Path(weights_path)
            if not weights_file.exists():
                print(f"Warning: Weights file not found at {weights_path}")
                # Try to find it in alternative locations
                alt_paths = [
                    Path(exp_dir / "yolo_cbam_scratch/train/train/weights/best.pt")
                ]
                
                for alt_path in alt_paths:
                    if alt_path.exists():
                        weights_path = str(alt_path)
                        print(f"Using alternative weights file: {weights_path}")
                        break
            
            if evaluate_model("yolo_cbam_scratch", weights_path, opt.batch, opt.device, exp_dir):
                visualize_results("yolo_cbam_scratch", eval_dir, exp_dir)
    
    # Print total execution time
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n\nExperiment completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Results saved to {exp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 experiments with different configurations")
    
    parser.add_argument("--experiment-name", type=str, default=f"exp_{time.strftime('%Y%m%d_%H%M%S')}", 
                        help="Name for this experiment run")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, 
                        help="Batch size for training and evaluation")
    parser.add_argument("--device", type=str, default="0", 
                        help="Device to use (e.g., '0', 'cpu')")
    
    # Model selection arguments
    parser.add_argument("--all", action="store_true", 
                        help="Run all model configurations")
    parser.add_argument("--yolo-pretrained", action="store_true", 
                        help="Run standard YOLOv8 with pretrained weights")
    parser.add_argument("--yolo-scratch", action="store_true", 
                        help="Run standard YOLOv8 from scratch")
    parser.add_argument("--yolo-cbam-scratch", action="store_true", 
                        help="Run YOLOv8 with CBAM attention from scratch")
    
    opt = parser.parse_args()
    
    # If no specific models selected, run all
    if not (opt.all or opt.yolo_pretrained or opt.yolo_scratch or opt.yolo_cbam_scratch):
        opt.all = True
    
    main(opt) 