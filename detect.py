from ultralytics import YOLO
import cv2
import math

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Initialize YOLO model
model = YOLO("best_ssaqta.pt")

# Object classes
classNames = ["flame"]

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (640, 480))

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Perform object detection
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # Draw bounding box if confidence is greater than or equal to 0.6
            if confidence >= 0.6:
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Class name
                cls = int(box.cls[0])

                # Object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

                print("Confidence --->", confidence)
                print("Class name -->", classNames[cls])

    # Display the frame on the screen
    cv2.imshow('Webcam', img)

    # Write the frame to the output video file
    out.write(img)

    # Check for 'q' key press to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and video writer
cap.release()
out.release()
cv2.destroyAllWindows()
