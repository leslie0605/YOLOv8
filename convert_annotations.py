import os
import json
import cv2
from tqdm import tqdm

def convert_bbox_to_yolo(img_width, img_height, bbox):
    """
    Convert bbox from JSON format (x_center, y_center, width, height) in pixels
    to YOLO format (x_center, y_center, width, height) normalized
    """
    x_center = bbox['coordinates']['x'] / img_width
    y_center = bbox['coordinates']['y'] / img_height
    width = bbox['coordinates']['width'] / img_width
    height = bbox['coordinates']['height'] / img_height
    
    # Ensure values are between 0 and 1
    x_center = min(max(x_center, 0), 1)
    y_center = min(max(y_center, 0), 1)
    width = min(max(width, 0), 1)
    height = min(max(height, 0), 1)
    
    return x_center, y_center, width, height

def convert_annotations(data_dir):
    """
    Convert annotations from JSON to YOLO format
    Args:
        data_dir: Base directory containing the dataset
    """
    # Class mapping from names to indices
    class_map = {'ch': 0, 'sc': 3, 'hdc': 2, 'dc': 1}  # Matching the order in data.yaml
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"Skipping {split} - directory not found")
            continue
            
        # Create labels directory if it doesn't exist
        labels_dir = os.path.join(split_dir, 'labels')
        os.makedirs(labels_dir, exist_ok=True)
        
        # Load JSON annotations
        json_path = os.path.join(split_dir, '_annotations.json')
        if not os.path.exists(json_path):
            print(f"Skipping {split} - annotations not found")
            continue
            
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        print(f"\nProcessing {split} set...")
        for ann in tqdm(annotations):
            image_filename = ann['image']
            image_path = os.path.join(split_dir, 'images', image_filename)
            
            # Read image dimensions
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Create corresponding txt filename
            txt_filename = os.path.splitext(image_filename)[0] + '.txt'
            txt_path = os.path.join(labels_dir, txt_filename)
            
            # Convert and write annotations
            with open(txt_path, 'w') as f:
                for bbox in ann['bboxes']:
                    # Get class index
                    class_idx = class_map[bbox['label']]
                    
                    # Convert bbox coordinates
                    x_center, y_center, width, height = convert_bbox_to_yolo(
                        img_width, img_height, bbox
                    )
                    
                    # Write to file in YOLO format
                    f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def main():
    # Base directory containing the dataset
    data_dir = 'data'
    
    print("Starting conversion from JSON to YOLO format...")
    convert_annotations(data_dir)
    print("\nConversion completed!")
    
    # Print summary of the conversion
    for split in ['train', 'valid', 'test']:
        labels_dir = os.path.join(data_dir, split, 'labels')
        if os.path.exists(labels_dir):
            num_labels = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
            print(f"{split} set: {num_labels} label files created")

if __name__ == '__main__':
    main() 