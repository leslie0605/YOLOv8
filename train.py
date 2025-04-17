import os
import argparse
import torch
from torch.optim import Adam
from model import create_model
from ultralytics import YOLO
from torch.utils.data import DataLoader

def main(args):
    # Set device
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Create our custom model with attention
    model = create_model(num_classes=args.num_classes)
    model = model.to(device)
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Create a base YOLO model for data loading and utilities
    base_model = YOLO('yolov8n.pt')
    
    # Train the model
    base_model.train(
        data=os.path.join(args.data_dir, 'data.yaml'),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch_size,
        device=args.device,
        workers=args.num_workers,
        project=args.checkpoint_dir,
        name='train',
        exist_ok=True,
        model=model  # Use our custom model
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Crochet Stitch Detector')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='runs', help='Directory to save checkpoints')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of stitch classes')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--device', type=str, default='', help='Device to train on (cpu or cuda:0)')
    
    args = parser.parse_args()
    main(args) 