To train the model:
python train.py --data_dir data

To evaluate the model:
yolo val model=runs/train/weights/best.pt data=data/data.yaml task=detect
