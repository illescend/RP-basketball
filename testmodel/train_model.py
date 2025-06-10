from ultralytics import YOLO
import torch

def get_device():
    """Automatically select devices -> mps（Mac） -> cpu"""
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device

# Select device for training
device = get_device()
print(device)
# # If there is no pre-trained model, use YOLO's default
PRE_TRAINED_MODEL = 'Yolo-Weights/yolov8n.pt'
# Load a model
model = YOLO(PRE_TRAINED_MODEL)
# Train the model
results = model.train(data='config.yaml', epochs=100, imgsz=640, device=device)