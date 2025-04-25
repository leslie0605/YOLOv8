# YOLOv8 with CBAM Attention

This project implements and compares YOLOv8 object detection models with and without CBAM (Convolutional Block Attention Module) attention mechanism. The implementation supports both training from scratch and using pretrained weights.

## Models

1. **Standard YOLOv8 (pretrained)** - YOLOv8 nano with pretrained weights
2. **Standard YOLOv8 (from scratch)** - YOLOv8 nano trained from scratch
3. **YOLOv8 with CBAM (from scratch)** - YOLOv8 nano with CBAM attention, trained from scratch

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- Ultralytics YOLO package

Install the required packages:

```bash
pip install ultralytics
pip install matplotlib seaborn scikit-learn
```

### Directory Structure

```
.
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
├── models/
│   ├── attention.py
│   └── yolov8n_cbam.yaml
├── outputs/                  # All output files are stored here
│   └── experiments/          # Experiment results
│       └── <experiment_name>/
│           ├── yolo_pretrained/
│           ├── yolo_scratch/
│           ├── yolo_cbam_scratch/
├── train.py
├── evaluate.py
├── visualize.py
├── run_experiments.py
└── README.md
```

## Step-by-Step Training and Evaluation

### Method 1: Automated Experiments with All Models

To train, evaluate, and visualize all model variants in one go:

Change the 'path' in data.yaml to the absolute path of your local machine's data folder.

```bash
python run_experiments.py --experiment-name my_experiment --epochs 50 --batch 16
```

This will:

1. Create an experiment directory structure in `outputs/experiments/my_experiment/`
2. Train all three model configurations
3. Evaluate each model on the test set
4. Generate visualizations

Options:

- `--experiment-name`: Name for this experiment (default: timestamp-based name)
- `--epochs`: Number of training epochs (default: 50, 200 is used in our experiment)
- `--batch`: Batch size for training and evaluation (default: 16)
- `--device`: Device to use (default: '0' for GPU, use 'cpu' for CPU training)

To run only specific model configurations, use these flags:

- `--yolo-pretrained`: Run standard YOLOv8 with pretrained weights
- `--yolo-scratch`: Run standard YOLOv8 from scratch
- `--yolo-cbam-scratch`: Run YOLOv8 with CBAM attention from scratch

### Method 2: Step-by-Step Manual Process

If you want more control, you can run each step manually.

#### 1. Training

**Standard YOLOv8 with Pretrained Weights**:

```bash
yolo train task=detect model=yolov8n.pt data=data/data.yaml epochs=50 batch=16 project=outputs/manual name=yolo_pretrained
```

**Standard YOLOv8 from Scratch**:

```bash
yolo train task=detect model=yolov8n.yaml data=data/data.yaml epochs=50 batch=16 project=outputs/manual name=yolo_scratch
```

**YOLOv8 with CBAM Attention from Scratch**:

```bash
python train.py --epochs=50 --batch=16 --device=0 --project=outputs/manual --name=yolo_cbam_scratch
```

#### 2. Evaluation (on test set)

```bash
python evaluate.py --weights=<path-to-best.pt-of-model> --batch=16 --device=0 --visualize=20 --plot-per-class --project=outputs/manual --name=yolo_cbam_scratch_eval
```

#### 3. Visualization

For any model, run the visualization script on the evaluation results directory:

```bash
python visualize.py --results-dir=outputs/manual/yolo_pretrained_eval --output-dir=outputs/manual/yolo_pretrained_viz
```

## Results

After running the experiments, you'll find detailed metrics and visualizations for each model, including:

- Precision-Recall curves
- Confusion matrices
- Detection examples
- Per-class performance charts

All results are organized in a consistent directory structure within the `outputs` folder.

## Implementation Details

- The CBAM attention module is implemented in `models/attention.py`
- The YOLOv8 model architecture with CBAM is defined in `models/yolov8n_cbam.yaml`
- The training process with weight transfer is handled in `train.py`
- The project uses a unified directory structure for all outputs in the `outputs` folder
