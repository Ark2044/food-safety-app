# AI-Powered Food Safety Detection

## Overview

The AI-Powered Food Safety Detection application leverages machine learning and computer vision technologies to analyze food safety by detecting harmful additives and evaluating ingredient safety. This web application provides users with an intuitive interface to upload images and input ingredients, ensuring that food products are safe and healthy.

## Features

- **Additive Detection**: Upload images of food items to detect harmful food additives and E-codes.
- **Ingredient Analysis**: Input a list of ingredients to analyze their safety and potential health risks.
- **Real-time Feedback**: Get immediate results and insights on food safety using advanced AI models.
- **User-Friendly Interface**: Navigate through the app easily to access essential food safety data.

## Technologies Used
- **Backend**: Python with Flask
- **Frontend**: HTML, TailwindCSS
- **Machine Learning**: OpenCV for image processing, EasyOCR for OCR
- **Data Analysis**: NumPy, Pandas

## Installation

### Prerequisites

Make sure you have Python installed on your machine. You can download it from [python.org](https://www.python.org/downloads/).

### Install Dependencies

Clone the repository and navigate to the project directory:

```bash
git clone <your-repo-url>
cd <your-project-folder>
```

### Install the required packages:

```bash
Copy code
pip install -r requirements.txt
```

### Install Tesseract-OCR
Install Tesseract-OCR by following the instructions for your operating system. You can find the installation guide [here](https://github.com/tesseract-ocr/tesseract).

### Usage
Start the Flask application:

```bash
python app.py
```

Open your web browser and go to http://127.0.0.1:5000 to access the application.

Choose an option to analyze food safety:

Additive Detection: Upload an image of the food item.
Ingredient Analysis: Enter a list of ingredients.