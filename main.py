"""
Face Detection & Filter App
----------------------------
Detects faces in a live webcam feed using OpenCV's Haar Cascade classifier
and applies a user-selected filter (box, blur, or grayscale) to each
detected face region in real time.

Author: <YOUR NAME HERE>
Course: Computer Vision - Flipped Course Evaluation
"""

import cv2
import argparse
import sys
import time


def load_face_detector():
    """Load OpenCV's pre-trained Haar Cascade face detector."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        print("Error: Could not load Haar Cascade classifier.")
        sys.exit(1)
    return detector


def apply_filter(frame, faces, filter_type):
    """
    Apply the chosen filter to every detected face region.

    frame       : the current video frame (numpy array)
    faces       : list of (x, y, w, h) rectangles from the detector
    filter_type : one of "box", "blur", "gray"
    """
    for (x, y, w, h) in faces:
        if filter_type == "box":
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        elif filter_type == "blur":
            face_roi = frame[y:y + h, x:x + w]
            blurred = cv2.GaussianBlur(face_roi, (35, 35), 30)
            frame[y:y + h, x:x + w] = blurred

        elif filter_type == "gray":
            face_roi = frame[y:y + h, x:x + w]
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            gray_face_3ch = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)
            frame[y:y + h, x:x + w] = gray_face_3ch

    return frame


def run(filter_type, camera_index, save_output):
    detector = load_face_detector()
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera index {camera_index}.")
        sys.exit(1)

    print("Face Filter App started.")
    print("Press 'q' to quit | Press 'b' box | 'l' blur | 'g' gray")

    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output.mp4", fourcc, 20.0, (640, 480))

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from camera.")
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        frame = apply_filter(frame, faces, filter_type)

        # FPS counter (basic performance metric for the report)
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}  Mode: {filter_type}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Face Detection & Filter App", frame)

        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            filter_type = "box"
        elif key == ord('l'):
            filter_type = "blur"
        elif key == ord('g'):
            filter_type = "gray"

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("App closed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time Face Detection & Filter App using OpenCV."
    )
    parser.add_argument(
        "--filter", type=str, default="box", choices=["box", "blur", "gray"],
        help="Filter to apply on detected faces (default: box)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index to use (default: 0)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save the output video as output.mp4"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.filter, args.camera, args.save)
