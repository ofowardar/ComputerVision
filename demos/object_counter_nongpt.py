# Import Libs
import cv2
from ultralytics import YOLO

# Load the Model
model = YOLO('yolov8n.pt')

# Take the frames.
cap = cv2.VideoCapture(0)
ret,frame = cap.read()
if not ret:
    print("Cannot read the frame!")
    exit()
