# Waste Classification and Management System
## Overview
This project focuses on classifying waste materials into biodegradable and non-biodegradable categories. The system identifies six types of waste: glass, metal, paper, plastic, biological waste, and battery, using advanced machine learning models and facilitates their classification in real-time. The goal is to promote effective waste management and recycling practices.

## Features
### Real-Time Waste Classification:
Classifies waste materials into six predefined categories using an AI-based model.

### Biodegradable & Non-Biodegradable Segregation:
Automatically segregates waste to streamline composting or recycling processes.

### User-Friendly Web Application:
A simple web interface where users can upload images of waste for classification.

### Business Strategy:
Biodegradable waste is converted to manure, while non-biodegradable waste is recycled or sold for reuse.

## Project Workflow
### Phase 1 - Model Development:
Data collection and preprocessing.
Dataset annotation using RoboFlow for bounding box annotations.
Model building and evaluation using CNN and classification techniques.

### Phase 2 - Web Application Development:
Backend setup using Flask for processing user inputs.
Integration of the trained model to classify waste from uploaded images.
User interface built with Bootstrap for styling.

## Dataset:
The dataset comprises 5300 images, annotated and categorized into six classes:
Glass
Metal
Paper
Plastic
Biological Waste
Battery
Data preprocessing and augmentation were conducted to improve model performance.

## Tech Stack
### Programming Language: Python
### Frameworks: Flask (backend), Bootstrap (frontend styling)
### Machine Learning: RoboFlow, CNN architecture for waste classification
### Tools: RoboFlow for dataset annotations

## Installation and Setup
### Prerequisites:
Python 3.7 or above
Pip package manager

## Steps to Run the Project
### Clone the repository:
git clone https://github.com/deepak12345678900/wasteclassifier.git
cd waste-classification

### Install the required dependencies:
pip install -r requirements.txt

### Run the Flask application:
python app.py

### Open the application in your browser:
http://127.0.0.1:5000/
Upload waste images for classification.


# Home Page to Upload Images
![Screenshot 2024-10-06 191827](https://github.com/user-attachments/assets/0847071b-2675-4fc4-b643-5d007a8fc915)

# Prediction Page to display results

![Screenshot 2024-10-06 193041](https://github.com/user-attachments/assets/1a3815d3-172a-43e8-897a-3471348f4691)

