# YOLOv8 with CBAM Attention

This project implements and compares YOLOv8 object detection models with and without CBAM (Convolutional Block Attention Module) attention mechanism. The implementation supports both training from scratch and using pretrained weights.

## Models

1. **Standard YOLOv8 (pretrained)** - YOLOv8 nano with pretrained weights
2. **Standard YOLOv8 (from scratch)** - YOLOv8 nano trained from scratch
3. **YOLOv8 with CBAM (pretrained)** - YOLOv8 nano with CBAM attention, using pretrained weights
4. **YOLOv8 with CBAM (from scratch)** - YOLOv8 nano with CBAM attention, trained from scratch

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
│           ├── yolo_cbam_pretrained/
│           ├── yolo_cbam_scratch/
│           └── summary_report.md
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
2. Train all four model configurations
3. Evaluate each model on the test set
4. Generate visualizations
5. Create a summary report comparing performance metrics

Options:

- `--experiment-name`: Name for this experiment (default: timestamp-based name)
- `--epochs`: Number of training epochs (default: 50)
- `--batch`: Batch size for training and evaluation (default: 16)
- `--device`: Device to use (default: '0' for GPU, use 'cpu' for CPU training)

To run only specific model configurations, use these flags:

- `--yolo-pretrained`: Run standard YOLOv8 with pretrained weights
- `--yolo-scratch`: Run standard YOLOv8 from scratch
- `--yolo-cbam-pretrained`: Run YOLOv8 with CBAM attention using pretrained weights
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

**YOLOv8 with CBAM Attention using Pretrained Weights**:

```bash
python train.py --epochs=50 --batch=16 --device=0 --pretrained --project=outputs/manual --name=yolo_cbam_pretrained
```

**YOLOv8 with CBAM Attention from Scratch**:

```bash
python train.py --epochs=50 --batch=16 --device=0 --project=outputs/manual --name=yolo_cbam_scratch
```

#### 2. Evaluation

**Standard YOLOv8 Models**:

```bash
yolo val task=detect model=outputs/manual/yolo_pretrained/weights/best.pt data=data/data.yaml split=test batch=16 save_json=True save_conf=True save=True project=outputs/manual name=yolo_pretrained_eval
```

**YOLOv8 with CBAM Attention**:

```bash
python evaluate.py --weights=outputs/manual/yolo_cbam_pretrained/weights/best.pt --batch=16 --device=0 --visualize=20 --plot-per-class --project=outputs/manual --name=yolo_cbam_pretrained_eval
```

#### 3. Visualization

For any model, run the visualization script on the evaluation results directory:

```bash
python visualize.py --results-dir=outputs/manual/yolo_pretrained_eval --output-dir=outputs/manual/yolo_pretrained_viz
```

## Advanced Analysis

For comparing multiple models side by side, you can use the automated experiment script with specific model flags. For example, to compare only the pretrained models:

```bash
python run_experiments.py --yolo-pretrained --yolo-cbam-pretrained --epochs 50
```

## Results

After running the experiments, you'll find:

1. A summary report in Markdown format comparing model performance (`outputs/experiments/<experiment_name>/summary_report.md`)
2. Detailed metrics and visualizations for each model, including:
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
- Data paths are relative, allowing the project to be used on different systems without modification
