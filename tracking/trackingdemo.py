import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("C:/Users/Ömer Faruk/Desktop/ComputerVision/videos/video1.mp4")

unique_person_ids = set()

while True:
    ret,frame = cap.read()

    if not ret:
        break

    results = model.track(frame,persist=True)

    for r in results:
        for box in r.boxes:

            if box.id is None:
                continue

            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            if label == "person" and conf > 0.5:
                track_id = int(box.id[0])
                unique_person_ids.add(track_id)

                x1,y1,x2,y2 = map(int,box.xyxy[0])

                cv2.rectangle(
                    frame,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    frame,
                    f"ID: {track_id}",

                    (x1,y1-10),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.6,
                    (255,0,0),
                    2
                )

    cv2.putText(
        frame,
        f"Unique Persons: {len(unique_person_ids)}",
        (10,40),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Tracking Demo",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()