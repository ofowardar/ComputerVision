from ultralytics import YOLO
import cv2 

# Load the model
model = YOLO("yolov8n.pt")

# Capture the video 
CAPTURE_PATH = "C:/Users/Ömer Faruk/Desktop/ComputerVision/videos/people.mp4"
cap = cv2.VideoCapture(CAPTURE_PATH)

if not cap.isOpened():
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, verbose=False)

    for r in results:
        for box in r.boxes:
            confidence = float(box.conf[0])
            if confidence < 0.5:
                continue

            class_id = int(box.cls[0])
            label = model.names[class_id]

            if label != "person":
                continue

            if box.id is None:
                continue

            track_id = int(box.id[0])

            x1,y1,x2,y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2,
            )

    cv2.imshow("Demo Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
