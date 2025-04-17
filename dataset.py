import os
import cv2
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml

def load_dataset_config(yaml_path='data/data.yaml'):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class CrochetStitchDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir (str): Base directory containing the data
            split (str): 'train', 'valid', or 'test'
            transform: Optional transform to be applied on a sample
        """
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.class_map = {'ch': 0, 'sc': 1, 'hdc': 2, 'dc': 3}
        
        # Load annotations
        with open(os.path.join(self.data_dir, '_annotations.json'), 'r') as f:
            self.annotations = json.load(f)
            
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # Get annotation data
        ann = self.annotations[idx]
        img_name = ann['image']
        
        # Load image
        img_path = os.path.join(self.data_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get bounding boxes and labels
        boxes = []
        labels = []
        
        for bbox in ann['bboxes']:
            # Get coordinates
            x = bbox['coordinates']['x']
            y = bbox['coordinates']['y']
            w = bbox['coordinates']['width']
            h = bbox['coordinates']['height']
            
            # Convert to x1, y1, x2, y2 format
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2
            
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_map[bbox['label']])
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply transformations
        if self.transform:
            if len(boxes) > 0:
                transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
                image = transformed['image']
                boxes = np.array(transformed['bboxes'])
                labels = np.array(transformed['class_labels'])
            else:
                transformed = self.transform(image=image)
                image = transformed['image']
        
        # Convert to tensors
        target = {
            'boxes': torch.FloatTensor(boxes) if len(boxes) > 0 else torch.zeros((0, 4)),
            'labels': torch.LongTensor(labels) if len(labels) > 0 else torch.zeros(0, dtype=torch.int64)
        }
        
        return image, target

def get_transform(train=True):
    """
    Get transformation pipeline
    """
    if train:
        transform = A.Compose([
            A.Resize(height=640, width=640),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    else:
        transform = A.Compose([
            A.Resize(height=640, width=640),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    return transform

def create_data_loaders(base_dir, batch_size=16, num_workers=4):
    """
    Create training and validation data loaders
    Args:
        base_dir (str): Base directory containing the data
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
    """
    # Create datasets
    train_dataset = CrochetStitchDataset(
        data_dir=base_dir,
        split='train',
        transform=get_transform(train=True)
    )
    
    val_dataset = CrochetStitchDataset(
        data_dir=base_dir,
        split='valid',
        transform=get_transform(train=False)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader

def collate_fn(batch):
    """
    Custom collate function to handle variable size batches
    """
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    return images, targets 