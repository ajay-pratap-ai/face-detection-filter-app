# Face Detection & Filter App

A real-time computer vision application that detects human faces in a webcam
feed (or a static image) using OpenCV's Haar Cascade classifier, and applies
one of three visual filters to each detected face: a bounding box, a Gaussian
blur, or a grayscale overlay.

## 1. Project Overview

- **Domain:** Computer Vision
- **Core technique:** Haar Cascade face detection (`cv2.CascadeClassifier`)
- **Features:**
  - Real-time face detection from a webcam
  - Three switchable filters: `box`, `blur`, `gray`
  - Live FPS counter overlay
  - Optional output video recording (`--save`)
  - Standalone image mode for machines without a webcam

## 2. Prerequisites

- Python 3.9 or higher
- A working webcam (only required for `main.py`; not needed for
  `detect_image.py`)

## 3. Setup Instructions

### Step 1: Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### Step 2: Create a virtual environment (recommended)
```bash
python -m venv venv

# Activate it:
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

## 4. Running the Project

### Option A — Live webcam mode
```bash
python main.py --filter box
```

Arguments:
| Flag        | Description                                   | Default |
|-------------|------------------------------------------------|---------|
| `--filter`  | `box`, `blur`, or `gray`                       | `box`   |
| `--camera`  | Camera index (useful if multiple cams exist)   | `0`     |
| `--save`    | Saves the session as `output.mp4`              | off     |

Controls while running:
- `b` → switch to box filter
- `l` → switch to blur filter
- `g` → switch to grayscale filter
- `q` → quit

### Option B — Static image mode (no webcam required)
```bash
python detect_image.py --input sample.jpg --filter blur --output result.jpg
```

This mode is provided so the project can be evaluated on machines without
camera access. Place any test image (with visible faces) in the project
folder and run the command above.

## 5. Project Structure
```
face-filter-app/
├── main.py            # Live webcam face detection & filtering
├── detect_image.py     # Static image face detection & filtering
├── requirements.txt    # Python dependencies
└── README.md
```

## 6. How It Works (Brief)

1. Each video frame (or the input image) is converted to grayscale, since
   Haar Cascade detection works on intensity gradients, not color.
2. `detectMultiScale()` scans the image at multiple scales to locate
   face-like regions, returning bounding boxes `(x, y, w, h)`.
3. The selected filter is applied only within each detected face region:
   - **box**: draws a rectangle and label around the face
   - **blur**: applies a strong Gaussian blur to anonymize the face
   - **gray**: desaturates only the face region
4. The processed frame is displayed (and optionally saved) in real time.

## 7. Known Limitations

- Haar Cascades can produce false positives/negatives in poor lighting or
  at extreme face angles.
- Performance depends on webcam resolution and CPU (no GPU acceleration
  used in this version).

## 8. Future Improvements

- Replace Haar Cascade with a DNN-based detector (e.g., OpenCV's
  `res10_300x300_ssd` model) for higher accuracy.
- Add emotion or age/gender estimation on top of detected faces.
