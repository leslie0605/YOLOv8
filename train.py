import os
import argparse
from model import create_model

def main(args):
    # Create model
    model = create_model(num_classes=args.num_classes)
    
    # Train the model using YOLOv8's native training method
    model.train(
        data=os.path.join(args.data_dir, 'data.yaml'),  # Path to data.yaml file
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch_size,
        device=args.device,
        workers=args.num_workers,
        project=args.checkpoint_dir,
        name='train',
        exist_ok=True
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