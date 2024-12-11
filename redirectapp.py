from flask import Flask, render_template, redirect, url_for, Response, jsonify
import cv2
from datetime import datetime
from ultralytics import YOLO
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
from collections import Counter

# Initialize Flask app
app = Flask(__name__)

# Load models
yolo_model = YOLO("yolov8n.pt")  # YOLOv8 pretrained model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Initialize webcam
camera = cv2.VideoCapture(0)

# Dummy data to store results
results = []
analysis_results = []

#loading freshness model
def load_freshness_model():
    model = YOLO("best.pt")  # Replace with your YOLO model path for freshness analysis
    return model

freshness_model = load_freshness_model()

# Helper functions for both apps (app.py and fruitapp.py)

# Detect and analyze function (from app.py)
def detect_and_analyze(frame):
    try:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = yolo_model.predict(frame, conf=0.5)
        detected_boxes = detections[0].boxes
        object_counts = {}

        for box in detected_boxes:
            class_id = int(box.cls[0])
            class_name = yolo_model.model.names[class_id]
            object_counts[class_name] = object_counts.get(class_name, 0) + 1

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "read brand details, packsize, brand name, and expiry date"}]}]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[pil_image], padding=True, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        output_ids = model.generate(**inputs, max_new_tokens=1024)
        output_text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        details = {"Brand": None, "Expiration Date": None}
        for line in output_text.split("\n"):
            if "Brand Name" in line:
                details["Brand"] = line.split(":")[1].strip()
            if "Expiration Date" in line:
                details["Expiration Date"] = line.split(":")[1].strip()

        timestamp = datetime.now().isoformat()
        if details["Expiration Date"]:
            expiration_date = datetime.strptime(details["Expiration Date"], "%d %b %Y")
            expired = expiration_date < datetime.now()
            life_span = (expiration_date - datetime.now()).days if not expired else None
        else:
            expired, life_span = None, None

        result = {
            "Timestamp": timestamp,
            "Brand": details["Brand"],
            "Expiry_Date": details["Expiration Date"],
            "Object_Counts": object_counts,
            "Expired": "Yes" if expired else "NA",
            "Expected_Life_Span_Days": life_span,
        }
        results.append(result)
    except Exception as e:
        print(f"Error in detection and analysis: {e}")

# Fruit freshness analysis function (from fruitapp.py)
def analyze_frame(frame):
    result = freshness_model(frame, conf=0.3, imgsz=640)
    detections = result[0].boxes
    output = []
    counts = Counter()

    for box in detections.data.tolist():
        x1, y1, x2, y2, confidence, class_id = box
        produce = freshness_model.names[int(class_id)]
        freshness = "Fresh" if "fresh" in produce.lower() else "Rotten"
        counts[produce] += 1

        output.append({
            "timestamp": datetime.now().isoformat(),
            "produce": produce,
            "freshness": freshness,
            "confidence": round(confidence * 100, 2),
            "bounding_box": {
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)
            }
        })

    return output, counts

# Video feed generator (for both apps)
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            detect_and_analyze(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_results = analyze_frame(frame_rgb)
            analysis_results.extend(current_results)

            results = yolo_model(frame)
            frame = results[0].plot()
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('mindex.html')

@app.route('/packaging_analysis')
def packaging_analysis():
    return redirect(url_for('index_page'))

@app.route('/fruit_freshness_analysis')
def fruit_freshness_analysis():
    return redirect(url_for('findex_page'))

@app.route('/index')
def index_page():
    return render_template('index.html')

@app.route('/findex')
def findex_page():
    counts = Counter()
    for result in analysis_results:
        counts[result['produce']] += 1
    return render_template('findex.html', results=analysis_results, counts=counts)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results', methods=['GET'])
def get_results():
    return jsonify(results)

@app.route('/fresults', methods=['GET'])
def get_fresults():
    return jsonify(analysis_results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
