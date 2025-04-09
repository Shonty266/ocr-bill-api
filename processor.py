import sys
import easyocr
import json
import os
import cv2
import uuid
import re
import hashlib
import contextlib
import io
from functools import wraps

# Suppress unwanted logs
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

# Try loading YOLO model (only when needed)
def load_yolo_model():
    try:
        from ultralytics import YOLO
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, 'Model', 'Bill_Detection.pt')
        return YOLO(model_path)
    except ImportError:
        return None

# Initialize EasyOCR only once
reader = easyocr.Reader(['en'])

def detect_and_crop_bills(image_path):
    """Detects and crops bills using YOLO or contour detection."""
    image = cv2.imread(image_path)
    if image is None:
        return []

    cropped_paths = []
    model = load_yolo_model()  # Load YOLO only when required

    if model:
        results = model(image)[0]
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())
            cropped_img = image[y1:y2, x1:x2]
            crop_name = f"{uuid.uuid4().hex[:8]}.jpg"
            crop_path = os.path.join("crops", crop_name)
            os.makedirs("crops", exist_ok=True)
            cv2.imwrite(crop_path, cropped_img)
            cropped_paths.append(crop_path)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 10000:
                x, y, w, h = cv2.boundingRect(cnt)
                cropped_img = image[y:y+h, x:x+w]
                crop_name = f"{uuid.uuid4().hex[:8]}.jpg"
                crop_path = os.path.join("crops", crop_name)
                os.makedirs("crops", exist_ok=True)
                cv2.imwrite(crop_path, cropped_img)
                cropped_paths.append(crop_path)

    return cropped_paths

def clean_text(text):
    """Cleans extracted text."""
    return ' '.join(''.join(char for char in text if char.isprintable()).split())

def extract_text_from_image(image_path):
    """Extracts text from a given image using EasyOCR."""
    try:
        text = reader.readtext(image_path, detail=0)
        return clean_text(" ".join(text))
    except Exception:
        return ""

def extract_fields(text):
    """Extracts bill-related fields from text."""
    if not text:
        return {'bill_number': 'UNKNOWN', 'subtotal': 0.0, 'date_of_issue': None, 'text': ''}

    text = clean_text(text)
    
    bill_number_match = re.search(r'(?:Bill|Invoice|Receipt|Ref)\s*(?:No\.?|#)?\s*([\w\-]+)', text, re.IGNORECASE)
    subtotal_match = re.search(r'(?:Total|Amount Due|Subtotal)[\s:₹$]*([\d,]+\.\d{1,2})', text, re.IGNORECASE)
    date_match = re.search(r'(?:Date|Invoice Date)[:\-\s]*([\d/.\-]+)', text, re.IGNORECASE)

    bill_number = bill_number_match.group(1) if bill_number_match else f"ID-{hashlib.md5(text.encode()).hexdigest()[:6]}"
    subtotal = float(subtotal_match.group(1).replace(',', '')) if subtotal_match else 0.0

    return {'bill_number': bill_number, 'subtotal': subtotal, 'date_of_issue': date_match.group(1) if date_match else None, 'text': text}

def process_images(input_path):
    """Processes images, detects bills, extracts text, and returns structured results."""
    if not os.path.exists(input_path):
        return {"error": "Path not found", "details": f"{input_path} does not exist"}

    results = {}

    def process_single_image(img_path, img_name):
        cropped_paths = detect_and_crop_bills(img_path)
        if not cropped_paths:
            results[img_name] = {"error": "No bills detected"}
            return

        for i, crop_path in enumerate(cropped_paths):
            text = extract_text_from_image(crop_path)
            fields = extract_fields(text)
            results[f"{img_name}_part{i+1}"] = fields
            os.remove(crop_path)  # Auto-delete cropped file

    if os.path.isfile(input_path) and input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        process_single_image(input_path, os.path.basename(input_path))
    elif os.path.isdir(input_path):
        for file_name in sorted(os.listdir(input_path)):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                process_single_image(os.path.join(input_path, file_name), file_name)
    else:
        return {"error": "Invalid file type", "details": "Must be an image or folder"}

    return results

if __name__ == "__main__":
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input path provided", "details": "Usage: python script.py <image_path>"}))
        sys.exit(1)

    input_path = sys.argv[1]

    try:
        output = json.dumps(process_images(input_path), indent=2)
        print(output)
    except Exception as e:
        print(json.dumps({"error": "Processing failed", "details": str(e)}, indent=2))
