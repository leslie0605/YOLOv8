# YOLOv8 with Attention for Crochet Stitch Detection

This project implements an enhanced version of YOLOv8 with attention mechanisms for detecting crochet stitches. The attention modules help the model focus on important patterns in crochet work, improving detection accuracy.

## Project Overview

Crochet stitch detection is challenging because of the repetitive patterns and similar appearances of different stitch types. This project addresses these challenges by:

1. Using YOLOv8 as the base object detection framework
2. Adding CBAM (Convolutional Block Attention Module) at strategic layers
3. Supporting both training from scratch and using pretrained weights

The attention mechanisms help the model focus on subtle differences between stitch types by enhancing important features in both channel and spatial dimensions.

## Project Structure

```
project/
├── data/                       # Dataset directory
│   ├── train/                  # Training images and labels
│   ├── valid/                  # Validation images and labels
│   ├── test/                   # Test images and labels
│   └── data.yaml               # Dataset configuration
├── models/
│   ├── attention.py            # CBAM attention module implementation
│   └── yolov8n_cbam.yaml       # Model architecture with attention
└── train.py                    # Training script
```

## Model Architecture

The model is based on YOLOv8-nano with CBAM attention modules inserted at three strategic locations:

1. After the initial feature extraction (64 channels)
2. After the middle layers (128 channels)
3. After deeper feature layers (256 channels)

Each CBAM module consists of:

- Channel attention that focuses on "which features are important"
- Spatial attention that focuses on "where the important features are located"

This dual attention approach is particularly effective for crochet stitch detection, where both the type of features and their spatial arrangement are important.

## Dataset

The dataset contains four types of crochet stitches:

- ch: Chain stitch
- dc: Double crochet
- hdc: Half double crochet
- sc: Single crochet

Images are annotated in YOLO format with bounding boxes around each stitch.

## Usage

### Prerequisites

```bash
pip install ultralytics
```

### Training

#### Train from scratch (no pretrained weights)

```bash
python train.py --epochs 100 --batch 16
```

#### Train with pretrained YOLOv8 weights

```bash
python train.py --epochs 100 --batch 16 --pretrained
```

#### Additional training options

```bash
python train.py --epochs 100 --batch 16 --pretrained --device 0 --workers 4 --verbose
```

Parameters:

- `--epochs`: Number of training epochs
- `--batch`: Batch size
- `--pretrained`: Use pretrained YOLOv8 weights
- `--device`: Device to train on (e.g., "0" for first GPU, empty for CPU)
- `--workers`: Number of data loading workers
- `--verbose`: Show detailed model information

### Evaluation

#### Evaluate the model

```bash
# For models trained from scratch
yolo val model=runs/cbam_scratch/weights/best.pt data=data/data.yaml task=detect

# For models trained with pretrained weights
yolo val model=runs/cbam_pretrained/weights/best.pt data=data/data.yaml task=detect
```

## Understanding the Results

The model performance is measured using standard object detection metrics:

- **Precision (P)**: Accuracy of positive predictions
- **Recall (R)**: Percentage of actual objects detected
- **mAP50**: Mean Average Precision at IoU=0.50
- **mAP50-95**: Mean Average Precision averaged over IoU thresholds from 0.50 to 0.95

These metrics are reported for each stitch type and for the overall model.

## Implementation Details

The CBAM attention module dynamically adapts to the channel dimensions of the layers it's attached to, making it compatible with different scales of the YOLOv8 model. The attention mechanism is implemented with:

1. Channel attention using global average pooling, followed by a bottleneck structure
2. Spatial attention using both average and max pooling operations
3. Residual connections to maintain gradient flow

When using pretrained weights, only the compatible layers are transferred, while the attention modules are initialized from scratch. This approach combines the feature extraction power of pretrained YOLOv8 with the enhanced focus provided by attention mechanisms.

## Acknowledgments

This project builds upon the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) implementation and incorporates the CBAM attention mechanism introduced in the paper "CBAM: Convolutional Block Attention Module" by Woo et al.
