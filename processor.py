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

# Decorator to suppress function output
def suppress_output(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return func(*args, **kwargs)
    return wrapper

# Redirect all output at startup
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

# Optional YOLO setup
try:
    from ultralytics import YOLO
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'Model', 'Bill_Detection.pt')  # 👈 updated path here
    model = YOLO(model_path)
    use_yolo = True
except ImportError:
    use_yolo = False

# Initialize EasyOCR reader with error handling
@suppress_output
def initialize_reader():
    try:
        return easyocr.Reader(['en'])
    except Exception as e:
        print(f"Error initializing EasyOCR: {str(e)}", file=sys.stderr)
        raise

reader = initialize_reader()

@suppress_output
def detect_and_crop_bills(image_path, output_dir="crops"):
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    cropped_paths = []

    if use_yolo:
        results = model(image)[0]
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())
            cropped_img = image[y1:y2, x1:x2]
            crop_name = f"{uuid.uuid4().hex[:8]}.jpg"
            crop_path = os.path.join(output_dir, crop_name)
            cv2.imwrite(crop_path, cropped_img)
            cropped_paths.append(crop_path)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4 and cv2.contourArea(cnt) > 10000:
                x, y, w, h = cv2.boundingRect(cnt)
                cropped = image[y:y+h, x:x+w]
                crop_name = f"{uuid.uuid4().hex[:8]}.jpg"
                crop_path = os.path.join(output_dir, crop_name)
                cv2.imwrite(crop_path, cropped)
                cropped_paths.append(crop_path)

    return cropped_paths

def clean_text(text):
    """Remove non-UTF-8 characters and excessive whitespace"""
    if not isinstance(text, str):
        text = str(text)
    # Remove non-printable characters but preserve common symbols
    cleaned = ''.join(char for char in text if char.isprintable() or char in '₹$€£.,:;-/\\')
    # Normalize whitespace
    return ' '.join(cleaned.split())

@suppress_output
def extract_text_from_image(image_path):
    try:
        text_results = reader.readtext(image_path, detail=0)
        combined_text = " ".join(text_results)
        return clean_text(combined_text)
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}", file=sys.stderr)
        return ""

def extract_fields(text):
    if not text:
        return {
            'bill_number': 'UNKNOWN',
            'subtotal': 0.0,
            'date_of_issue': None,
            'text': ''
        }

    # Clean and normalize text before processing
    text = clean_text(text)
    
    # Improved regex patterns
    bill_number_match = re.search(
        r'(?:Bill|Invoice|Ref|Order|Receipt)\s*(?:Number|#|No\.?)?\s*[:\-]?\s*([A-Za-z0-9\-]+)',
        text, re.IGNORECASE
    )
    
    subtotal_match = re.search(
        r'(?:Total|Grand\s*Total|Subtotal|Amount\s*Due|Fare|Payable|Amt\.?)[\s|:\-=₹$]*\s?([0-9,]+(?:\.\d{1,2})?)',
        text, re.IGNORECASE
    )
    
    date_match = re.search(
        r'(?:Date|Invoice Date|Issued On|Purchase Date|Bill Date)[\s|:\-]*'
        r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\d{4}[/\-\.]\d{2}[/\-\.]\d{2}|'
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})',
        text, re.IGNORECASE
    )

    bill_number = bill_number_match.group(1) if bill_number_match else None

    if not bill_number:
        vendor_match = re.search(
            r'(?:Vendor|Company|Merchant|Retailer|Store|Shop)\s*[:\-]?\s*([A-Za-z0-9\s&]+)',
            text, re.IGNORECASE
        )
        vendor_name = vendor_match.group(1).strip() if vendor_match else text[:20]
        bill_number = vendor_name.upper() + '-' + hashlib.md5(text.encode()).hexdigest()[:6]

    try:
        subtotal = float(subtotal_match.group(1).replace(',', '')) if subtotal_match else 0.0
    except (ValueError, AttributeError):
        subtotal = 0.0

    return {
        'bill_number': bill_number,
        'subtotal': subtotal,
        'date_of_issue': date_match.group(1) if date_match else None,
        'text': text
    }

def process_images(input_path):
    results_to_send = {}

    if not os.path.exists(input_path):
        return {"error": "Path not found", "details": f"The path {input_path} does not exist"}

    def process_single_image(img_path, img_name):
        try:
            cropped_paths = detect_and_crop_bills(img_path)
            if not cropped_paths:
                results_to_send[img_name] = {
                    "error": "No bills detected",
                    "details": "The image processing didn't find any bill-like regions"
                }
                return

            for i, crop_path in enumerate(cropped_paths):
                try:
                    text = extract_text_from_image(crop_path)
                    fields = extract_fields(text)
                    results_to_send[f"{img_name}_part{i+1}"] = fields
                    # Clean up temporary crop file
                    os.remove(crop_path)
                except Exception as e:
                    results_to_send[f"{img_name}_part{i+1}_error"] = {
                        "error": "Processing failed",
                        "details": str(e)
                    }

        except Exception as e:
            results_to_send[img_name] = {
                "error": "Image processing failed",
                "details": str(e)
            }

    if os.path.isfile(input_path) and input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        file_name = os.path.basename(input_path)
        process_single_image(input_path, file_name)

    elif os.path.isdir(input_path):
        for file_name in sorted(os.listdir(input_path)):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_path, file_name)
                process_single_image(image_path, file_name)

    else:
        return {"error": "Invalid file type or path", "details": "The input must be an image file or directory containing images"}

    return results_to_send

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input path provided", "details": "Usage: python script.py <image_path>"}))
        sys.exit(1)

    input_path = sys.argv[1]
    
    try:
        final_result = process_images(input_path)
    except Exception as e:
        final_result = {
            "error": "Unexpected error",
            "details": str(e)
        }

    # Restore stdout and print only the JSON
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    
    try:
        output = json.dumps(final_result, ensure_ascii=True, indent=2)
        print(output)
    except Exception as e:
        error_output = {
            "error": "JSON encoding failed",
            "details": str(e),
            "original_data": str(final_result)
        }
        print(json.dumps(error_output, ensure_ascii=True, indent=2))