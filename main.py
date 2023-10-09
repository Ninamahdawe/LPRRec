import cv2
import os
import pytesseract
from ultralytics import YOLO

# Get the current working directory
projectdir = os.getcwd()

# Define the paths to YOLO weights and OpenVINO model
yolo_weights_path = os.path.join(projectdir, "testfiles", "best.pt")
openvino_model_path = os.path.join(
    projectdir, "testfiles", "best_openvino_model")

try:
    # Load YOLO model
    yolo_model = YOLO(yolo_weights_path)
    print("YOLO Model Loaded")
except FileNotFoundError:
    print(f"Error: File not found at path: {yolo_weights_path}")

try:
    # Load OpenVINO model
    ov_model = YOLO(openvino_model_path)
    print("OpenVINO Model Loaded")
except FileNotFoundError:
    print(f"Error: File not found at path: {openvino_model_path}")

# Open a video capture
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 640))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.merge((frame, frame, frame))

    try:
        # Process frame with YOLO model
        yolo_results = yolo_model(frame)
        yolo_boxes = yolo_results[0].boxes.xyxy

        for box in yolo_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 0), 2)

            # Extract text using Pytesseract
            roi = frame[int(y1):int(y2), int(x1):int(x2)]
            text = pytesseract.image_to_string(roi, config='--psm 6')
            print(f"YOLO Text: {text}")
    except NameError:
        pass  # yolo_model is not defined

    try:
        # Process frame with OpenVINO model
        ov_results = ov_model(frame)
        ov_boxes = ov_results[0].boxes.xyxy

        for box in ov_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (255, 0, 0), 2)

            # Extract text using Pytesseract
            roi = frame[int(y1):int(y2), int(x1):int(x2)]
            text = pytesseract.image_to_string(roi, config='--psm 6')
            print(f"OpenVINO Text: {text}")
    except NameError:
        pass  # ov_model is not defined

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()
