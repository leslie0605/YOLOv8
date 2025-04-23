import argparse
import torch
from pathlib import Path
from ultralytics import YOLO
from models.attention import register_attention  # noqa


def main(opt):
    register_attention() 
    
    if opt.pretrained:
        print("Using pretrained YOLOv8n as base model")
        
        base_model = YOLO('yolov8n.pt')
        
        model = YOLO(Path("models/yolov8n_cbam.yaml"))
        
        pretrained_dict = base_model.model.state_dict()
        model_dict = model.model.state_dict()
        
        cbam_indices = []
        for i, m in enumerate(model.model.model):
            if hasattr(m, '__class__') and m.__class__.__name__ == 'CBAM':
                cbam_indices.append(i)
        
        print(f"Found {len(cbam_indices)} CBAM attention layers at indices: {cbam_indices}")
        

        transfer_dict = {}
        mismatched_layers = []
        cbam_related_layers = 0
        
        for k, v in pretrained_dict.items():
            if 'detect' in k:
                continue
                
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    if any(f'model.{i}.' in k for i in cbam_indices):
                        cbam_related_layers += 1
                        transfer_dict[k] = v
                    else:
                        transfer_dict[k] = v
                else:
                    mismatched_layers.append(f"{k}: pretrained {v.shape} vs model {model_dict[k].shape}")
        
        print(f"Transferred {len(transfer_dict)}/{len(model_dict)} layers")
        print(f"Including {cbam_related_layers} layers related to CBAM attention")
        
        if mismatched_layers:
            print(f"Skipped {len(mismatched_layers)} layers due to shape mismatch:")
            for layer in mismatched_layers[:5]:  # 仅显示前几个不匹配的层
                print(f"  {layer}")
            if len(mismatched_layers) > 5:
                print(f"  ...and {len(mismatched_layers)-5} more")
        
        model.model.load_state_dict(transfer_dict, strict=False)
        print("Successfully transferred compatible weights")
        
        if opt.verbose:
            print("\nModel structure with attention layers:")
            for i, (name, module) in enumerate(model.model.named_modules()):
                if 'CBAM' in str(module.__class__):
                    print(f"  {i}: {name} - {module.__class__.__name__}")
    else:
        print("Training from scratch")
        model = YOLO(Path("models/yolov8n_cbam.yaml"))
    
    # Use project and name from command line args if provided
    project = opt.project if opt.project else "runs"
    name = opt.name if opt.name else "cbam_" + ("pretrained" if opt.pretrained else "scratch")
    
    model.train(
        data=Path("data/data.yaml"),
        epochs=opt.epochs,
        imgsz=640,
        batch=opt.batch,
        device=opt.device or None,
        workers=opt.workers,
        project=project,
        name=name,
        exist_ok=True,
        pretrained=False,  
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained YOLOv8n weights")
    parser.add_argument("--verbose", action="store_true", help="Show detailed model information")
    parser.add_argument("--project", type=str, help="Project directory for saving results")
    parser.add_argument("--name", type=str, help="Experiment name")
    main(parser.parse_args())