from flask import Flask, request, jsonify
import os
from processor import process_images

app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    file_path = os.path.join("temp_upload", image_file.filename)
    os.makedirs("temp_upload", exist_ok=True)
    image_file.save(file_path)

    result = process_images(file_path)

    # Clean up
    os.remove(file_path)

    return jsonify(result)

@app.route("/", methods=["GET"])
def home():
    return "Bill OCR API Running!"

if __name__ == "__main__":
    app.run(debug=True)
