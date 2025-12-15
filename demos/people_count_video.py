import cv2
from ultralytics import YOLO


# Load a pre-trained YOLO model
model = YOLO("yolov8n.pt")

# Open a video file or capture device
video_path = "C:/Users/Ömer Faruk/Desktop/ComputerVision/videos/video1.mp4"
cap = cv2.VideoCapture(video_path)


# Control to frame
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Take frame and process

person_frame_counter = 0 

while True:
    ret, frame = cap.read()
    if not ret:
        break


    # Process the frame with the YOLO model
    results = model(frame, verbose=False)

    person_count = 0

    # Detection 
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label == "person":
                person_count += 1
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                # Draw Bounding box
                xyxy = box.xyxy[0]
                conf = box.conf[0]
                cv2.rectangle(
                    frame,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2

                )
    # Display person count
    cv2.putText(
        frame,
        f"Person Count: {person_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    if person_count > 0:
        person_frame_counter += 1
    else:
        person_frame_counter = 0

    if person_frame_counter >= 3:
        h, w, _ = frame.shape
        cv2.putText(
            frame,
            "PERSON DETECTED",
            (w - 270, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
    )

    # Display the frame
    cv2.imshow("People Count", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
