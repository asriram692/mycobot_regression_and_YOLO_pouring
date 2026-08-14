"""Train YOLOv8 object detection model on custom dataset for target cup detection."""
from ultralytics import YOLO

model = YOLO("yolov8s.pt") 

model.train(
    data="cup-detection.v3i.yolov8/data.yaml",
    epochs=10,         
    imgsz=320,         
    device="mps",      
    workers=2          
)