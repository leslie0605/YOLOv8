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

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets):
        # Unpack predictions
        pred_boxes = predictions[..., :4]  # [batch, anchors, 4]
        pred_obj = predictions[..., 4:5]   # [batch, anchors, 1]
        pred_cls = predictions[..., 5:]    # [batch, anchors, num_classes]
        
        # Calculate losses
        box_loss = self.mse(pred_boxes, targets['boxes'])
        obj_loss = self.bce(pred_obj, targets['obj'])
        cls_loss = self.bce(pred_cls, targets['cls'])
        
        # Combine losses
        total_loss = box_loss + obj_loss + cls_loss
        
        return {
            'box_loss': box_loss,
            'obj_loss': obj_loss,
            'cls_loss': cls_loss,
            'total_loss': total_loss
        }

class CrochetStitchDetector(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Load pretrained YOLOv8 model
        self.yolo = YOLO('yolov8n.pt').model
        
        # Freeze backbone
        for param in self.yolo.model[:10].parameters():
            param.requires_grad = False
            
        # Add pattern attention modules after specific YOLO layers
        self.pattern_attention1 = PatternAttentionModule(256)  # After layer 4
        self.pattern_attention2 = PatternAttentionModule(512)  # After layer 6
        self.pattern_attention3 = PatternAttentionModule(1024) # After layer 9
        
        # Modify the detection head for our number of classes
        self.yolo.model[-1].nc = num_classes
        
        # Add loss function
        self.loss_fn = YOLOLoss()
        
    def forward(self, x, targets=None):
        # Get intermediate features from YOLO
        features = []
        for i, module in enumerate(self.yolo.model):
            x = module(x)
            if i in [4, 6, 9]:
                features.append(x)
            
        # Apply pattern attention to intermediate features
        features[0] = self.pattern_attention1(features[0])
        features[1] = self.pattern_attention2(features[1])
        features[2] = self.pattern_attention3(features[2])
        
        # Get predictions from YOLO head
        predictions = self.yolo.model[-1](features[-1])
        
        # If in training mode, calculate and return loss
        if self.training and targets is not None:
            return self.loss_fn(predictions, targets)
            
        return predictions

def create_model(num_classes, pretrained=True):
    """
    Create and initialize the model
    """
    # Load a pretrained YOLO model
    model = YOLO('yolov8n.pt')
    
    # Modify the model for our number of classes
    model.model.nc = num_classes
    
    # Train the model
    return model 