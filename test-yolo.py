from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Load a pretrained model
model = YOLO('yolo11s-seg.pt')

image_path = 'doraemon.jpg'
results = model(image_path)

masks = results[0].masks
coords = masks.xy
print(coords[0])
