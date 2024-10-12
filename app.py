# import os
# import numpy as np
# import pandas as pd
# import cv2
# import pytesseract
# import re
# from flask import Flask, render_template, request, redirect, url_for

# app = Flask(__name__)

# # Load the dataset
# data = pd.read_csv("Harmful foods India.csv")

# # Preprocess the data to create a list of additives and E-codes
# additives = data['Additive Name'].str.upper().tolist()  # Ensure all additives are uppercase
# e_codes = data['E-Code'].dropna().tolist()  # Create a list of E-codes

# # Function to preprocess the image for better OCR results
# def preprocess_image(image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     new_width = 800
#     aspect_ratio = gray_image.shape[1] / gray_image.shape[0]
#     new_height = int(new_width / aspect_ratio)
#     resized_image = cv2.resize(gray_image, (new_width, new_height))
#     contrast_image = cv2.convertScaleAbs(resized_image, alpha=1.5, beta=0)
#     blurred_image = cv2.GaussianBlur(contrast_image, (5, 5), 0)
#     thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY, 11, 2)
#     kernel = np.ones((3, 3), np.uint8)
#     morphed_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#     return morphed_image

# # Function to extract text from an image using OpenCV and Tesseract
# def extract_text_from_image(image):
#     processed_image = preprocess_image(image)
#     custom_config = r'--oem 3 --psm 6'
#     text = pytesseract.image_to_string(processed_image, config=custom_config)
#     cleaned_text = re.sub(r'[^A-Za-z0-9\s,]', '', text).upper()
#     detected_words = [word.strip() for word in cleaned_text.split() if word.strip()]
#     return detected_words, text

# # Mock AI functions (replace with actual models)
# def analyze_food_quality(image):
#     detected_e_codes = []  # Replace with actual detection logic
#     if image is not None:  # Simulate detecting E-codes
#         detected_e_codes = ["E211", "E621"]  # Example detected E-codes
#     return detected_e_codes

# def predict_disease(ingredients):
#     disease_risks = []
#     if ingredients:  # Simulate disease prediction based on unsafe ingredients
#         if "E211" in ingredients:
#             disease_risks.append("Increased risk of allergies")
#         if "E621" in ingredients:
#             disease_risks.append("Potential for high blood pressure")
#     return disease_risks

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/additive_detection', methods=['GET', 'POST'])
# def additive_detection():
#     if request.method == 'POST':
#         # Handle image upload
#         if 'image' in request.files:
#             file = request.files['image']
#             if file:
#                 # Convert the image to OpenCV format
#                 in_memory_file = np.frombuffer(file.read(), np.uint8)
#                 image = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)

#                 # Extract text from the image
#                 detected_additives, raw_text = extract_text_from_image(image)

#                 harmful_status = []

#                 if detected_additives:
#                     # Check for harmful additives or E-codes in the image
#                     for item in detected_additives:
#                         if item in e_codes:
#                             additive_row = data.loc[data['E-Code'] == item]
#                             if not additive_row.empty:
#                                 additive_name = additive_row['Additive Name'].values[0]
#                                 harmfulness = additive_row['Harmfulness'].values[0]
#                                 harmful_status.append(f"{additive_name} (E-code: {item}): {harmfulness}")
#                         elif item in additives:
#                             level = data.loc[data['Additive Name'].str.upper() == item, 'Harmfulness'].values[0]
#                             harmful_status.append(f"{item}: {level}")

#                     # AI Quality Analysis
#                     detected_e_codes = analyze_food_quality(image)
#                     if detected_e_codes:
#                         for code in detected_e_codes:
#                             harmful_status.append(f"Detected harmful substance: {code}")

#                     # Disease Prediction
#                     health_risks = predict_disease(detected_additives)
#                     if health_risks:
#                         for risk in health_risks:
#                             harmful_status.append(f"Potential health risk: {risk}")

#                 return render_template('result.html', raw_text=raw_text, harmful_status=harmful_status)
#     return render_template('additive_detection.html')

# @app.route('/ingredient_analysis', methods=['GET', 'POST'])
# def ingredient_analysis():
#     if request.method == 'POST':
#         actual_ingredients_input = request.form['ingredients']
#         actual_ingredients = [ingredient.strip().upper() for ingredient in actual_ingredients_input.split(',')]
#         safety_issues = []

#         for ingredient in actual_ingredients:
#             if ingredient in additives:
#                 level = data.loc[data['Additive Name'].str.upper() == ingredient, 'Harmfulness'].values[0]
#                 safety_issues.append(f"{ingredient}: {level}")
#             elif ingredient in e_codes:
#                 additive_row = data.loc[data['E-Code'] == ingredient]
#                 if not additive_row.empty:
#                     additive_name = additive_row['Additive Name'].values[0]
#                     harmfulness = additive_row['Harmfulness'].values[0]
#                     safety_issues.append(f"{additive_name} (E-code: {ingredient}): {harmfulness}")
#             else:
#                 safety_issues.append(f"{ingredient} is safe.")

#         return render_template('ingredient_analysis.html', safety_issues=safety_issues)

#     return render_template('ingredient_analysis.html', safety_issues=[])

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host='0.0.0.0', port=port)


import os
import numpy as np
import pandas as pd
import cv2
import re
from flask import Flask, render_template, request
import easyocr  # Import EasyOCR
from difflib import get_close_matches  # For similarity checking

app = Flask(__name__)

# Load the EasyOCR reader globally
reader = easyocr.Reader(['en'])  # Specify the languages you want to use

# Load the dataset
data = pd.read_csv("Harmful foods India.csv")

# Preprocess the data to create a list of additives and E-codes, normalize to uppercase
additives = data['Additive Name'].str.upper().tolist()
e_codes = data['E-Code'].dropna().tolist()

# Function to check allowed file types
def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Function to preprocess the image for better OCR results
def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise the image
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)

    # Resize image while maintaining aspect ratio
    new_width = 800
    aspect_ratio = denoised_image.shape[1] / denoised_image.shape[0]
    new_height = int(new_width / aspect_ratio)
    resized_image = cv2.resize(denoised_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Increase contrast and brightness
    contrast_image = cv2.convertScaleAbs(resized_image, alpha=1.5, beta=30)

    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(contrast_image, -1, kernel)

    return sharpened_image

def extract_text_from_image(image):
    processed_image = preprocess_image(image)

    # Use EasyOCR to detect text
    results = reader.readtext(processed_image)

    # Combine the detected text from EasyOCR results
    cleaned_text = " ".join([result[1] for result in results])
    cleaned_text = re.sub(r'[^A-Za-z0-9\s,]', '', cleaned_text).upper()  # Clean text and convert to upper case
    detected_words = [word.strip() for word in cleaned_text.split() if word.strip()]
    return detected_words, cleaned_text

def analyze_food_quality(detected_additives):
    detected_items = []
    for item in detected_additives:
        item_upper = item.upper()
        
        # Check if the exact additive or E-code exists
        if item_upper in additives:
            detected_items.append(item_upper)
        elif item_upper in e_codes:
            detected_items.append(item_upper)
        else:
            # Check for similar matches with existing additives
            similar_additives = get_close_matches(item_upper, additives, n=1, cutoff=0.6)
            if similar_additives:
                detected_items.append(similar_additives[0])  # Use the closest match

    return detected_items

def get_harmfulness_status(detected_items):
    harmful_status = []
    for item in detected_items:
        item = item.strip()  # Remove leading/trailing spaces

        # Check if it's an E-code
        if item.startswith("E"):
            additive_row = data.loc[data['E-Code'] == item]
            if not additive_row.empty:
                additive_name = additive_row['Additive Name'].values[0]
                harmfulness = additive_row['Harmfulness'].values[0]
                harmful_status.append(f"{additive_name} (E-code: {item}): {harmfulness}")
            else:
                harmful_status.append(f"{item} (E-code): Unknown harmfulness")
        else:
            # Normalize item for comparison
            additive_row = data.loc[data['Additive Name'].str.strip().str.upper() == item]
            if not additive_row.empty:
                harmfulness = additive_row['Harmfulness'].values[0]
                harmful_status.append(f"{item}: {harmfulness}")
            else:
                harmful_status.append(f"{item} is not in the dataset.")

    return harmful_status

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/additive_detection', methods=['GET', 'POST'])
def additive_detection():
    if request.method == 'POST':
        # Handle image upload
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                # Convert the image to OpenCV format
                in_memory_file = np.frombuffer(file.read(), np.uint8)
                image = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)

                # Extract text from the image
                detected_additives, raw_text = extract_text_from_image(image)

                harmful_status = []

                if detected_additives:
                    # Check for harmful additives or E-codes in the image
                    detected_items = analyze_food_quality(detected_additives)
                    harmful_status = get_harmfulness_status(detected_items)

                # Construct output message based on whether harmful additives were found
                if harmful_status:
                    result_message = "Found the following harmful additives or E-codes:"
                else:
                    result_message = "No harmful additives or E-codes found."

                return render_template('result.html', raw_text=raw_text, harmful_status=harmful_status, result_message=result_message)
            else:
                return render_template('additive_detection.html', error="Invalid file format. Please upload an image.")
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
                    safety_issues.append(f"{ingredient} (E-code): Unknown harmfulness")
            else:
                safety_issues.append(f"{ingredient} is safe.")

        return render_template('ingredient_analysis.html', safety_issues=safety_issues)

    return render_template('ingredient_analysis.html', safety_issues=[])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
