import argparse
import torch
from pathlib import Path
from ultralytics import YOLO
from models.attention import register_attention  # noqa


def main(opt):
    register_attention() 
    
    # 先加载预训练的YOLOv8n模型
    if opt.pretrained:
        print("Using pretrained YOLOv8n as base model")
        
        # 加载官方预训练模型
        base_model = YOLO('yolov8n.pt')
        
        # 创建包含attention的模型
        model = YOLO(Path("models/yolov8n_cbam.yaml"))
        
        # 获取预训练模型和新模型的状态字典
        pretrained_dict = base_model.model.state_dict()
        model_dict = model.model.state_dict()
        
        # 1. 首先分析模型结构，找出所有CBAM层的索引
        cbam_indices = []
        for i, m in enumerate(model.model.model):
            if hasattr(m, '__class__') and m.__class__.__name__ == 'CBAM':
                cbam_indices.append(i)
        
        print(f"Found {len(cbam_indices)} CBAM attention layers at indices: {cbam_indices}")
        
        # 2. 只保留名称匹配且形状相同的层
        transfer_dict = {}
        mismatched_layers = []
        cbam_related_layers = 0
        
        for k, v in pretrained_dict.items():
            # 跳过检测头部分
            if 'detect' in k:
                continue
                
            # 如果目标模型有相同名称的层
            if k in model_dict:
                # 比较形状是否相同
                if v.shape == model_dict[k].shape:
                    # 检查这个层是否是CBAM层或在CBAM之后
                    if any(f'model.{i}.' in k for i in cbam_indices):
                        cbam_related_layers += 1
                        # 我们仍然转移它，但做个记录
                        transfer_dict[k] = v
                    else:
                        # 常规层，直接转移
                        transfer_dict[k] = v
                else:
                    mismatched_layers.append(f"{k}: pretrained {v.shape} vs model {model_dict[k].shape}")
        
        # 打印统计信息
        print(f"Transferred {len(transfer_dict)}/{len(model_dict)} layers")
        print(f"Including {cbam_related_layers} layers related to CBAM attention")
        
        if mismatched_layers:
            print(f"Skipped {len(mismatched_layers)} layers due to shape mismatch:")
            for layer in mismatched_layers[:5]:  # 仅显示前几个不匹配的层
                print(f"  {layer}")
            if len(mismatched_layers) > 5:
                print(f"  ...and {len(mismatched_layers)-5} more")
        
        # 3. 加载筛选后的预训练权重
        model.model.load_state_dict(transfer_dict, strict=False)
        print("Successfully transferred compatible weights")
        
        # 4. 打印最终模型的结构，确认CBAM层的位置
        if opt.verbose:
            print("\nModel structure with attention layers:")
            for i, (name, module) in enumerate(model.model.named_modules()):
                if 'CBAM' in str(module.__class__):
                    print(f"  {i}: {name} - {module.__class__.__name__}")
    else:
        print("Training from scratch")
        model = YOLO(Path("models/yolov8n_cbam.yaml"))
    
    model.train(
        data=Path("data/data.yaml"),
        epochs=opt.epochs,
        imgsz=640,
        batch=opt.batch,
        device=opt.device or None,
        workers=opt.workers,
        project="runs",
        name="cbam_" + ("pretrained" if opt.pretrained else "scratch"),
        exist_ok=True,
        pretrained=False,  # 我们手动处理预训练，所以这里保持False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained YOLOv8n weights")
    parser.add_argument("--verbose", action="store_true", help="Show detailed model information")
    main(parser.parse_args())