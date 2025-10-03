# IADA-201-2405744--zivantikasingh

SmartWasteAI - Computer Vision-based Waste Segregation System
1. Project Overview
This project, 

SmartWasteAI, develops an intelligent image-based waste classification system for EcoLogic Solutions, a smart city tech startup. The primary goal is to address the major challenge of 


ineffective waste segregation at collection centers, which hinders recycling and increases landfill waste.


system automatically identifies the type of waste from an uploaded image and suggests the correct color-coded disposal bin, thereby helping people sort their waste correctly and responsibly.


Key Features Implemented:

Waste Detection and Classification: Utilizes deep learning models to accurately classify waste as biodegradable, recyclable, or hazardous.


Bin Recommendation: Based on the classification, the system suggests the correct color-coded disposal bin.




Confidence Scoring: Displays the prediction's confidence score to support user decision-making.



Interactive User Interface: A clean, user-friendly web interface built with Streamlit.

2. Methodology and Data Preparation
2.1.
Research and Requirements (

Problem: Reducing human error in waste sorting, improving recycling efficiency, and benefiting the environment.


Waste Categories: The system is designed to classify waste into three main types: biodegradable (e.g., food waste), recyclable (e.g., paper/plastic), and hazardous (e.g., batteries/medical waste).


Bin Coding: The system recommends bins based on standard color codes: Green for biodegradable, Blue for recyclable, and Red for hazardous waste.

I/O Definition:


Input: An image of the waste.


Output: Waste type, correct bin color, and confidence score.


2.2.
Data Collection and Preprocessing (

Dataset Source: The provided dataset link was used, containing images categorized into 10 classes (e.g., plastic, glass, metal, shoes).


Dataset Subset: A subset of at least 100 images per class was selected for use.

Preprocessing Steps:


Resizing: All selected images were resized to a fixed dimension of 224×224 pixels to ensure uniform input to the neural network.


Augmentation: Image augmentation methods (e.g., rotating, flipping, zooming, adjusting brightness) were applied to make the model more robust.

Splitting: The final dataset was split into:


Training: 70% 


Validation: 15% 


Testing: 15% 


Organization: The dataset was structured in separate folders for each class (e.g., /plastic, /metal) as required by machine learning libraries.

3. Model Development and Training
3.1.
Models Used (
The project uses a two-step approach for high accuracy and efficiency:


Object Detection: YOLOv5 was used to detect the waste object in the image, leveraging its speed for object identification.


Classification: A Convolutional Neural Network (CNN), specifically MobileNet or EfficientNet (depending on implementation choice), was used to classify the detected object into one of the three final categories.

3.2. Training Parameters and Techniques
Training Epochs: [10 epochs]

4. Performance Metrics and Results
The model's performance was assessed using standard metrics:


Export to Sheets
Summarised Results ()

The model achieved a test accuracy of [80%], demonstrating strong performance in distinguishing between the three main waste types. The 

confusion matrix revealed  that the model occasionally confused 


5. Deployment and Accessibility
5.1. Streamlit Web App Demo
The web application is 

neat and responsive, allowing non-technical users to easily upload an image and receive a classification, confidence score, and bin recommendation.



