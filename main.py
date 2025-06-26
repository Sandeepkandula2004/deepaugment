from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import requests

# Initialize Flask app
app = Flask(__name__)


#model = load_model("model.h5")

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to predict tumor type
import requests

# Replace this function in your Flask app
def predict_tumor(image_path):
    api_url = "https://Sandeep2004-myspace.hf.space/predict"
    with open(image_path, "rb") as f:
        response = requests.post(api_url, files={"file": f})
    
    if response.status_code == 200:
        data = response.json()
        label = data["prediction"]
        confidence = data["confidence"]
        return (label if label == "No Tumor" else f"Tumor: {label}"), confidence
    else:
        return "Error contacting prediction server", 0.0


# Route for the main page (index.html)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        if file:
            # Save the file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            # Predict the tumor
            result, confidence = predict_tumor(file_location)

            # Return result along with image path for display
            return render_template('index.html', result=result, confidence=f"{confidence*100:.2f}%", file_path=f'/uploads/{file.filename}')

    return render_template('index.html', result=None)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)


