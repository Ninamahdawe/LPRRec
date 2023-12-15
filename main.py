import cv2
import numpy as np
import time
# from ultralytics import YOLO
import pytesseract


def detect_and_recognize_license_plate(image):
    """Detects and recognizes the license plate in an image.

    Args:
        image: A NumPy array representing the image.

    Returns:
        A string representing the license plate number, or None if no license plate
        was detected or recognized.
    """

    start_time = time.time()

    # Perform object detection using YOLO
    yolo_model = YOLO("license_plate_openvino_model", task="detect")
    yolo_results = yolo_model(image)

    # Check if YOLO detected any objects
    if hasattr(yolo_results, 'xyxy') and yolo_results.xyxy is not None and len(yolo_results.xyxy[0]) > 0:
        # Assuming the first detected object is the license plate
        license_plate_box = yolo_results.xyxy[0][0].cpu().numpy()

        # Crop the license plate from the image
        license_plate_image = image[
            int(license_plate_box[1]):int(license_plate_box[3]),
            int(license_plate_box[0]):int(license_plate_box[2])
        ]

        # Preprocess the license plate image
        license_plate_image = cv2.resize(license_plate_image, (300, 100))
        license_plate_image = cv2.cvtColor(
            license_plate_image, cv2.COLOR_BGR2GRAY)
        license_plate_image = cv2.GaussianBlur(license_plate_image, (3, 3), 0)

        # Perform OCR on the license plate image
        try:
            license_plate_number = pytesseract.image_to_string(
                license_plate_image, lang='eng', config='--psm 6').strip()
        except Exception as e:
            print('Error performing OCR:', e)
            return None

        # Draw a rectangle around the license plate
        cv2.rectangle(image, (int(license_plate_box[0]), int(license_plate_box[1])),
                      (int(license_plate_box[2]), int(license_plate_box[3])), (0, 255, 0), 2)

        # Add text to the image
        cv2.putText(image, license_plate_number,
                    (int(license_plate_box[0]), int(
                        license_plate_box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Display the image with a delay
        cv2.imshow('Image', image)
        cv2.waitKey(0)

        # Close all OpenCV windows
        cv2.destroyAllWindows()

        end_time = time.time()
        detection_time = end_time - start_time
        print('Detection time:', detection_time, 'seconds')

        return license_plate_number
    else:
        print('No license plate detected.')
        return None


def main():
    image = cv2.imread('testfiles/test.jpg')

    if image is not None:
        license_plate_number = detect_and_recognize_license_plate(image)

        if license_plate_number is not None:
            print('License plate number:', license_plate_number)
        else:
            print('License plate not recognized.')
    else:
        print('Error: Image not found.')


if __name__ == '__main__':
    main()
