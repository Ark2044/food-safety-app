import numpy as np
import pandas as pd
import cv2
import pytesseract
import re
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("Harmful foods India.csv")

# Preprocess the data to create a list of additives and E-codes
additives = data['Additive Name'].str.upper().tolist()  # Ensure all additives are uppercase
e_codes = data['E-Code'].dropna().tolist()  # Create a list of E-codes

# Function to preprocess the image for better OCR results
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    new_width = 800
    aspect_ratio = gray_image.shape[1] / gray_image.shape[0]
    new_height = int(new_width / aspect_ratio)
    resized_image = cv2.resize(gray_image, (new_width, new_height))
    contrast_image = cv2.convertScaleAbs(resized_image, alpha=1.5, beta=0)
    blurred_image = cv2.GaussianBlur(contrast_image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    morphed_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morphed_image

# Function to extract text from an image using OpenCV and Tesseract
def extract_text_from_image(image):
    processed_image = preprocess_image(image)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed_image, config=custom_config)
    cleaned_text = re.sub(r'[^A-Za-z0-9\s,]', '', text).upper()
    detected_words = [word.strip() for word in cleaned_text.split() if word.strip()]
    return detected_words, text

# Mock AI functions (replace with actual models)
def analyze_food_quality(image):
    detected_e_codes = []  # Replace with actual detection logic
    if image is not None:  # Simulate detecting E-codes
        detected_e_codes = ["E211", "E621"]  # Example detected E-codes
    return detected_e_codes

def predict_disease(ingredients):
    disease_risks = []
    if ingredients:  # Simulate disease prediction based on unsafe ingredients
        if "E211" in ingredients:
            disease_risks.append("Increased risk of allergies")
        if "E621" in ingredients:
            disease_risks.append("Potential for high blood pressure")
    return disease_risks

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/additive_detection', methods=['GET', 'POST'])
def additive_detection():
    if request.method == 'POST':
        # Handle image upload
        if 'image' in request.files:
            file = request.files['image']
            if file:
                # Convert the image to OpenCV format
                in_memory_file = np.frombuffer(file.read(), np.uint8)
                image = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)

                # Extract text from the image
                detected_additives, raw_text = extract_text_from_image(image)

                harmful_status = []

                if detected_additives:
                    # Check for harmful additives or E-codes in the image
                    for item in detected_additives:
                        if item in e_codes:
                            additive_row = data.loc[data['E-Code'] == item]
                            if not additive_row.empty:
                                additive_name = additive_row['Additive Name'].values[0]
                                harmfulness = additive_row['Harmfulness'].values[0]
                                harmful_status.append(f"{additive_name} (E-code: {item}): {harmfulness}")
                        elif item in additives:
                            level = data.loc[data['Additive Name'].str.upper() == item, 'Harmfulness'].values[0]
                            harmful_status.append(f"{item}: {level}")

                    # AI Quality Analysis
                    detected_e_codes = analyze_food_quality(image)
                    if detected_e_codes:
                        for code in detected_e_codes:
                            harmful_status.append(f"Detected harmful substance: {code}")

                    # Disease Prediction
                    health_risks = predict_disease(detected_additives)
                    if health_risks:
                        for risk in health_risks:
                            harmful_status.append(f"Potential health risk: {risk}")

                return render_template('result.html', raw_text=raw_text, harmful_status=harmful_status)
    return render_template('additive_detection.html')

@app.route('/ingredient_analysis', methods=['GET', 'POST'])
def ingredient_analysis():
    if request.method == 'POST':
        actual_ingredients_input = request.form['ingredients']
        actual_ingredients = [ingredient.strip().upper() for ingredient in actual_ingredients_input.split(',')]
        safety_issues = []

        for ingredient in actual_ingredients:
            if ingredient in additives:
                level = data.loc[data['Additive Name'].str.upper() == ingredient, 'Harmfulness'].values[0]
                safety_issues.append(f"{ingredient}: {level}")
            elif ingredient in e_codes:
                additive_row = data.loc[data['E-Code'] == ingredient]
                if not additive_row.empty:
                    additive_name = additive_row['Additive Name'].values[0]
                    harmfulness = additive_row['Harmfulness'].values[0]
                    safety_issues.append(f"{additive_name} (E-code: {ingredient}): {harmfulness}")
            else:
                safety_issues.append(f"{ingredient} is safe.")

        return render_template('ingredient_analysis.html', safety_issues=safety_issues)

    return render_template('ingredient_analysis.html', safety_issues=[])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
