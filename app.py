from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import shutil
from processor import process_images

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins (allow frontend to access this API)

UPLOAD_FOLDER = "temp_upload"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    return "✅ Bill OCR API Running!"

@app.route("/process", methods=["POST"])
def process():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    file_path = os.path.join(UPLOAD_FOLDER, image.filename)

    image.save(file_path)

    try:
        # Call your OCR processing logic
        result = process_images(file_path)
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    finally:
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify(result)

if __name__ == "__main__":
    print("🚀 Bill OCR API running at: http://localhost:5001")
    app.run(debug=True, host="0.0.0.0", port=5001)
