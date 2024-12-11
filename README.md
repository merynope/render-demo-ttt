# render-demo
# Packaging and Fruit Freshness Analysis Flask App

This project is a Flask-based web application designed for real-time packaging analysis and fruit freshness detection using computer vision. It runs on an NVIDIA Jetson TX1 and integrates with a conveyor belt system to hold products for camera module display, providing video feed to the Flask app. The app uses object detection models (YOLOv8 and custom models) to analyze packaging details and detect fruit freshness.

## Features

- **Packaging Analysis**: Detects packaging details such as brand name, pack size, and expiry date using the Qwen2 model, along with object detection to count the number of items detected in a frame.
- **Fruit Freshness Analysis**: Uses YOLOv8 to detect fruits and evaluate their freshness based on confidence scores. Provides real-time analysis of different fruit types and estimates their lifespan.
- **Conveyor Belt Integration**: A conveyor belt holds products for the camera module, which captures video feed for real-time analysis.
- **Video Stream**: Live video feed from the camera is streamed and processed for packaging and freshness analysis.

## Setup and Requirements

### Hardware Requirements:
- **NVIDIA Jetson TX1** (or compatible NVIDIA Jetson device)
- **Webcam or Camera Module** (for capturing live video feed)
- **Conveyor Belt** (for holding products for display)
- **Internet Connection** (for downloading models and dependencies)

### Software Requirements:
- **Python 3.x**
- **Flask** (for web app framework)
- **OpenCV** (for video capture and image processing)
- **PyTorch** (for deep learning models)
- **YOLOv8** (for object detection)
- **Transformers** (for Qwen2 model and processor)
- **PIL** (for image handling)
- **Ultralytics** (for YOLOv8)

### Steps to run the file

1. Clone the repository:
   ```bash
   git clone https://github.com/merynope/render-demo.git
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Linux/macOS
   venv\Scripts\activate  # On Windows
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
4. Install additional dependencies for NVIDIA Jetson TX1:
   ```bash
   sudo apt-get install libopencv-dev python-opencv
5. run the flask app
   ```bash
   python redirectapp.py

