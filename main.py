# 📂 Import necessary modules
import os  # Handles file paths and directory operations
import sys  # Allows modification of the system path to import local modules

# 💡 Add the project's root directory to the system path
# This ensures that Python can locate and import the "YoloDetectorWindowv4" module
sys.path.append(os.path.abspath(os.getcwd()))

# 🧐 Import the YOLO object detection module
import YoloDetectorWindowv4

# 🏠 Define the local working directory
# Retrieve the current working directory and ensure the path format is consistent
chemin = os.getcwd().replace("\\", "/") + "/"

# 📸 Perform object detection on the specified image
# The image "OK-immeuble-haussmannien.jpg" is processed using YOLO detection
YoloDetectorWindowv4.detector(chemin + "/OK-immeuble-haussmannien.jpg")
