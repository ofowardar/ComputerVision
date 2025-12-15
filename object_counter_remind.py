import cv2
from ultralytics import YOLO 

model = YOLO('yolov8n.pt')

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Can not open the webcam!")
    exit()

while True:
    ret,frame = capture.read()

    if not ret:
        print("Can not take the frame!!")
        break

    results = model(frame)

    object_counter = 0

    for r in results:
        for box in r.boxes:
            object_counter+= 1
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = model.names[class_id]

            if confidence < 0.5:
                continue

            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2)
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                f"{label}---{confidence:.2f}",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,0,0),
                1
            )