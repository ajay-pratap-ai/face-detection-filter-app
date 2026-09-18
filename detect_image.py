"""
Face Detection on a Static Image
---------------------------------
Same detection logic as main.py, but works on a single input image
instead of a live webcam feed. Useful for testing on machines without
a camera, and for generating result screenshots for the project report.

Usage:
    python detect_image.py --input sample.jpg --filter blur --output result.jpg
"""

import cv2
import argparse
import sys
import os


def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        print("Error: Could not load Haar Cascade classifier.")
        sys.exit(1)
    return detector


def apply_filter(frame, faces, filter_type):
    for (x, y, w, h) in faces:
        if filter_type == "box":
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        elif filter_type == "blur":
            roi = frame[y:y + h, x:x + w]
            frame[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (35, 35), 30)
        elif filter_type == "gray":
            roi = frame[y:y + h, x:x + w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            frame[y:y + h, x:x + w] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Detect and filter faces in an image.")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--filter", type=str, default="box",
                         choices=["box", "blur", "gray"], help="Filter type")
    parser.add_argument("--output", type=str, default="result.jpg",
                         help="Path to save the output image")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    image = cv2.imread(args.input)
    if image is None:
        print("Error: Could not read the image file. Check the format.")
        sys.exit(1)

    detector = load_face_detector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    print(f"Detected {len(faces)} face(s) in '{args.input}'.")

    result = apply_filter(image, faces, args.filter)
    cv2.imwrite(args.output, result)
    print(f"Result saved to '{args.output}'.")


if __name__ == "__main__":
    main()
