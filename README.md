# Window Detection with YOLO

## Overview
This project uses a YOLO-based neural network to detect windows on building facades. The model takes an image as input and returns the same image with bounding boxes and labels indicating detected windows.

## Features
- Uses a custom dataset for training.
- Outputs images with detected windows highlighted.
- Utilizes OpenCV and YOLO for object detection.

## Installation
Ensure you have Python installed, then install the required dependencies:

```sh
pip install -r requirements.txt
```

## Dependencies
The project requires the following libraries:
- `numpy`
- `opencv-python`
- `matplotlib`
- `scikit-image`

## Usage
Run the following command to detect windows in an image:

```sh
python main.py
```

The script processes an image (`OK-immeuble-haussmannien.jpg` by default) and outputs a version with detected windows.

## Model and Configuration
- The YOLO model is configured with:
  - **Configuration file:** `configuration/window_yolov4.cfg`
  - **Weights file:** `model/window_yolov4_last.weights`
  - **Label file:** `label/window.names`

## How It Works
1. The image is read and preprocessed.
2. The YOLO model detects objects.
3. Bounding boxes are drawn around detected windows.
4. The processed image is displayed and saved.

## Customization
- Modify `main.py` to change the input image.
- Adjust confidence thresholds in `YoloDetectorWindowv4.py` if needed.

## Contributions
Feel free to contribute! Open an issue or submit a pull request with improvements or bug fixes.

