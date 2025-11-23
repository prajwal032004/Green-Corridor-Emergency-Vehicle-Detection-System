from ultralytics import YOLO
model=YOLO('best.pt')  # load a pretrained YOLOv8n model
results=model(source=0,show=True,conf=0.4,save=True)  # run inference on the webcam