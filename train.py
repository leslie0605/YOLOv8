import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from tqdm import tqdm
from dataset import create_data_loaders
from model import create_model
import torch.nn as nn
from torch.optim import Adam
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from evaluate import evaluate

def get_model(num_classes):
    # Load a pre-trained model
    backbone = torchvision.models.resnet50(weights='DEFAULT')
    
    # Remove the last two layers (avgpool and fc)
    backbone = nn.Sequential(*list(backbone.children())[:-2])
    
    # FasterRCNN needs to know the number of output channels in the backbone
    backbone.out_channels = 2048
    
    # Define anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    # Create FasterRCNN model
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes + 1,  # +1 for background
        rpn_anchor_generator=anchor_generator,
        min_size=640,
        max_size=640
    )
    
    return model

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    
    for images, targets in tqdm(data_loader, desc='Training'):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    
    return total_loss / len(data_loader)

def validate(model, val_loader, device):
    model.train()  # Set to train mode to get losses
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validation'):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # During validation, we need to pass targets to get losses
            loss_dict = model(images, targets)
            
            # Handle both dictionary and list outputs
            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
            else:
                losses = sum(loss_dict)  # If it's a list, sum directly
            
            total_loss += losses.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        args.img_dir,
        args.label_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    model = get_model(num_classes=args.num_classes)
    model = model.to(device)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best_model.pt'))
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
            
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Crochet Stitch Detector')
    parser.add_argument('--img_dir', type=str, required=True, help='Path to image directory')
    parser.add_argument('--label_dir', type=str, required=True, help='Path to label directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of stitch classes')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    
    args = parser.parse_args()
    main(args) 