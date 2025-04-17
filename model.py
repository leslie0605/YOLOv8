from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatternAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Generate pattern queries, keys, and values
        query = self.conv1(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.conv2(x).view(batch_size, -1, H * W)
        value = self.conv3(x).view(batch_size, -1, H * W)
        
        # Calculate attention scores
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # Residual connection
        return self.gamma * out + x

class CrochetStitchDetector(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Load pretrained YOLOv8 model
        self.yolo = YOLO('yolov8n.pt')
        
        # Freeze backbone
        for param in self.yolo.model.model[:10].parameters():
            param.requires_grad = False
            
        # Add pattern attention modules after specific YOLO layers
        self.pattern_attention1 = PatternAttentionModule(256)  # After layer 4
        self.pattern_attention2 = PatternAttentionModule(512)  # After layer 6
        self.pattern_attention3 = PatternAttentionModule(1024) # After layer 9
        
        # Modify the detection head for our number of classes
        self.yolo.model.model[-1].nc = num_classes
        
    def forward(self, x):
        # Get intermediate features from YOLO
        features = []
        for i, module in enumerate(self.yolo.model.model):
            x = module(x)
            if i in [4, 6, 9]:
                features.append(x)
            
        # Apply pattern attention to intermediate features
        features[0] = self.pattern_attention1(features[0])
        features[1] = self.pattern_attention2(features[1])
        features[2] = self.pattern_attention3(features[2])
        
        # Continue with YOLO detection head
        return self.yolo.model.model[-1](features[-1])

def create_model(num_classes, pretrained=True):
    """
    Create and initialize the model
    """
    model = CrochetStitchDetector(num_classes=num_classes, pretrained=pretrained)
    return model 