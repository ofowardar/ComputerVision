from ultralytics import YOLO
import cv2 


MODEL_NAME = "yolov8n.pt"
VIDEO_PATH = "C:/Users/Ömer Faruk/Desktop/ComputerVision/videos/arac.mp4"
LINE_Y = 300
CONF_THRESHOLD = 0.5 


model = YOLO(MODEL_NAME)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    exit()


#Tracking Memory 
counted_ids = set()
total_count = 0 

while True:
    ret,frame = cap.read()
    if not ret:
        print("cant take frame!")
        break
    results = model.track(frame,persist=True)

    # Draw counting line
    cv2.line(
        frame,
        (0,LINE_Y),
        (frame.shape[1],LINE_Y),
        (0,0,255),
        2
    )

    for r in results:
        for box in r.boxes:
            if box.id is None:
                continue
            confidence = float(box.conf[0])
            if confidence < CONF_THRESHOLD:
                continue
            class_id = int(box.cls[0])
            label = model.names[class_id]

            # only count the people
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            track_id = int(box.id[0])

            #Bounding box

            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                f"ID {track_id}, Label: {label}",
                (x1,y1-8),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0,255,0),
                2
            )

            # Center point of object
            center_y = int((y1+y2) /2 )

            #Line crossing logic
            if center_y > LINE_Y and track_id not in counted_ids:
                counted_ids.add(track_id)
                total_count += 1

    cv2.putText(
        frame,
        f"Total Count: {total_count}",
        (20,40),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0,255,255),
        2
    )
    cv2.imshow("Line Crossing Counter",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()