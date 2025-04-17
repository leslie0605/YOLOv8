import os
import torch
import argparse
from tqdm import tqdm
import numpy as np
import cv2
from dataset import create_data_loaders, load_dataset_config
from model import create_model
from torchvision.ops import box_iou
from collections import defaultdict

class StitchEvaluator:
    def __init__(self, num_classes, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        self.total_detections = 0
        self.total_ground_truth = 0
        self.class_metrics = {}
        
        # Initialize per-class detections for mAP calculation
        for i in range(self.num_classes):
            self.class_metrics[i] = {
                'detections': [],  # List of [confidence, is_true_positive]
                'num_gt': 0
            }
    
    def update(self, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        # Update ground truth counts
        if len(gt_boxes) > 0:
            for gt_label in gt_labels:
                self.class_metrics[gt_label.item()]['num_gt'] += 1
        
        if len(pred_boxes) == 0:
            return
        
        # If no ground truth, all predictions are false positives
        if len(gt_boxes) == 0:
            for pred_idx in range(len(pred_boxes)):
                pred_label = pred_labels[pred_idx].item()
                pred_score = pred_scores[pred_idx].item()
                self.class_metrics[pred_label]['detections'].append([pred_score, 0])
            return
        
        # Calculate IoU between predictions and ground truth
        iou_matrix = box_iou(pred_boxes, gt_boxes)
        
        # Track which ground truth boxes have been matched
        matched_gt = set()
        
        # Sort predictions by confidence score
        scores_sorted, indices = torch.sort(pred_scores, descending=True)
        
        for idx in indices:
            pred_label = pred_labels[idx].item()
            pred_score = pred_scores[idx].item()
            
            # Find best matching ground truth for this prediction
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gt:
                    continue
                    
                if gt_labels[gt_idx].item() != pred_label:
                    continue
                    
                iou = iou_matrix[idx, gt_idx]
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # If we found a match above the threshold
            if best_iou > self.iou_threshold:
                self.class_metrics[pred_label]['detections'].append([pred_score, 1])
                matched_gt.add(best_gt_idx)
            else:
                self.class_metrics[pred_label]['detections'].append([pred_score, 0])
    
    def compute_ap(self, detections, num_gt):
        """
        Compute Average Precision for a single class
        """
        if num_gt == 0:
            return 0.0
            
        if len(detections) == 0:
            return 0.0
            
        # Sort detections by confidence
        detections = sorted(detections, key=lambda x: x[0], reverse=True)
        
        # Initialize precision/recall points
        precisions = []
        recalls = []
        true_positives = 0
        false_positives = 0
        
        # Calculate precision and recall at each detection
        for _, is_true_positive in detections:
            if is_true_positive:
                true_positives += 1
            else:
                false_positives += 1
                
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / num_gt
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Convert to numpy arrays
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        
        # Compute AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        return ap
    
    def compute(self):
        # Compute mAP
        aps = []
        for class_id, metrics in self.class_metrics.items():
            ap = self.compute_ap(metrics['detections'], metrics['num_gt'])
            aps.append(ap)
        
        mAP = np.mean(aps)
        
        # Compute per-class metrics
        class_results = {}
        for class_id, metrics in self.class_metrics.items():
            detections = metrics['detections']
            num_gt = metrics['num_gt']
            
            if len(detections) == 0:
                precision = 0
                recall = 0
                f1_score = 0
            else:
                true_positives = sum(1 for _, is_tp in detections if is_tp)
                false_positives = sum(1 for _, is_tp in detections if not is_tp)
                
                precision = true_positives / (true_positives + false_positives + 1e-6)
                recall = true_positives / (num_gt + 1e-6)
                f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
            
            class_results[class_id] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'AP': aps[class_id]
            }
        
        return {
            'overall': {
                'mAP@50': mAP
            },
            'per_class': class_results
        }

def visualize_predictions(image, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, output_path):
    """
    Visualize predictions and ground truth on the image
    """
    image = image.cpu().permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    
    # Draw ground truth boxes in green
    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box.cpu().numpy()
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f'GT: {label}', (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw prediction boxes in blue
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = box.cpu().numpy()
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(image, f'Pred: {label} ({score:.2f})', (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def compute_ap(recalls, precisions):
    # Compute Average Precision using the 11-point interpolation
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap = ap + p / 11.
    return ap

def evaluate(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    
    # Initialize metrics
    metrics_per_class = defaultdict(lambda: {'tp': [], 'fp': [], 'scores': [], 'num_gt': 0})
    
    # Collect all predictions and targets
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc='Evaluating'):
            # Stack images into a batch tensor
            images = torch.stack(images).to(device)
            
            # Get predictions
            outputs = model(images)  # Shape: (batch_size, num_anchors, 5 + num_classes)
            
            # Process each image in the batch
            for idx, (output, target) in enumerate(zip(outputs, targets)):
                # Get ground truth boxes and labels
                gt_boxes = target['boxes'].to(device)
                gt_labels = target['labels'].to(device)
                
                # Update ground truth counts
                for label in gt_labels:
                    metrics_per_class[label.item()]['num_gt'] += 1
                
                # Get predictions for this image
                # Each row in output is: [x1, y1, x2, y2, obj_conf, class_scores...]
                pred_boxes = output[:, :4]  # Get boxes in x1,y1,x2,y2 format
                obj_conf = output[:, 4]  # Object confidence
                class_scores = output[:, 5:]  # Class scores
                
                # Get class predictions and scores
                class_conf, class_pred = class_scores.max(1)  # Get max confidence and corresponding class
                pred_scores = obj_conf * class_conf  # Combine object and class confidence
                pred_labels = class_pred
                
                if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                    continue
                
                # Compute IoU between predictions and ground truth
                iou_matrix = box_iou(pred_boxes, gt_boxes)
                
                # Process each prediction
                gt_matched = set()
                for pred_idx in range(len(pred_boxes)):
                    pred_label = pred_labels[pred_idx].item()
                    pred_score = pred_scores[pred_idx].item()
                    
                    # Find best matching ground truth box
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx in range(len(gt_boxes)):
                        if gt_idx in gt_matched:
                            continue
                            
                        if gt_labels[gt_idx].item() != pred_label:
                            continue
                            
                        iou = iou_matrix[pred_idx, gt_idx]
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    # Store prediction result
                    metrics = metrics_per_class[pred_label]
                    metrics['scores'].append(pred_score)
                    
                    if best_iou >= iou_threshold and best_gt_idx not in gt_matched:
                        metrics['tp'].append(1)
                        metrics['fp'].append(0)
                        gt_matched.add(best_gt_idx)
                    else:
                        metrics['tp'].append(0)
                        metrics['fp'].append(1)
    
    # Compute AP for each class
    ap_per_class = {}
    mean_ap = 0
    
    for class_id, metrics in metrics_per_class.items():
        if metrics['num_gt'] == 0:
            continue
            
        # Convert to numpy arrays
        scores = np.array(metrics['scores'])
        tp = np.array(metrics['tp'])
        fp = np.array(metrics['fp'])
        
        # Sort by score
        indices = np.argsort(-scores)
        tp = tp[indices]
        fp = fp[indices]
        
        # Compute cumulative precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / metrics['num_gt']
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Append sentinel values
        recalls = np.concatenate(([0.], recalls, [1.]))
        precisions = np.concatenate(([0.], precisions, [0.]))
        
        # Compute AP
        ap = compute_ap(recalls, precisions)
        ap_per_class[class_id] = ap
        mean_ap += ap
    
    mean_ap = mean_ap / len(ap_per_class) if ap_per_class else 0
    
    return {
        'mAP@50': mean_ap,
        'AP@50_per_class': ap_per_class
    }

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset config
    config = load_dataset_config()
    class_names = config['names']
    
    # Create data loader (we only need validation)
    _, val_loader = create_data_loaders(
        base_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Load model
    model = create_model(num_classes=len(class_names))
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Evaluate
    metrics = evaluate(model, val_loader, device, args.iou_threshold)
    
    # Print results
    print("\nOverall Metrics:")
    print(f"mAP@50: {metrics['mAP@50']:.4f}")
    
    print("\nPer-class Metrics:")
    for class_id, ap in metrics['AP@50_per_class'].items():
        print(f"\nClass {class_id} ({class_names[class_id]}):")
        print(f"AP@50: {ap:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Crochet Stitch Detector')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='Directory to save visualization results')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for positive detection')
    
    args = parser.parse_args()
    main(args) 