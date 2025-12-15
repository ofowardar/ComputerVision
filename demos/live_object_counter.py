# Import Necessary Libraries
import cv2
from ultralytics import YOLO


# Import the model
model = YOLO('yolov8n.pt')

# Activate the webcam
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Can not open the webcam!")
    exit()

# Main Program
while True:
    ret, frame = capture.read()
    if not ret:
        print("Can not take the frame!!")
        break

    # Prediction with YOLO
    results = model(frame)

    #base counter
    object_counter = 0

    #Process the object which detected
    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = model.names[class_id]

            # Threshold for conf
            if confidence < 0.5:
                continue

            object_counter+=1

            # Draw the boxes
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            overlay = frame.copy()

            cv2.rectangle(overlay, (0, 0), (350, 60), (0, 0, 0), -1)
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            # Write the label
            cv2.putText(
                frame,
                f"Total Objects: {object_counter}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
)

            cv2.imshow("Live Object Counter",frame)

            # Exit the program with "q"
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


capture.relase()
cv2.destroyAllWindows()