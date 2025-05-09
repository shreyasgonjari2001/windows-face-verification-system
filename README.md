Face Verification System
This project implements a face verification system using OpenCV for image processing, Haar Cascade Classifier for face detection, and a Random Forest model for identity verification.

Key Features
Face Detection:

Utilizes Haar Cascade Classifier from OpenCV to detect faces in real-time via webcam or images.

Feature Extraction & Verification:

Extracted face regions are converted to feature vectors (e.g., using raw pixel values or embeddings).

Trained a Random Forest Classifier to verify if two faces belong to the same person.

Face Matching:

System compares an input face with stored reference data to authenticate identities.

Can be extended to support multiple known faces and unknown face rejection.

Tech Stack
Python (OpenCV, NumPy, Scikit-learn)

Haar Cascade for face detection

Random Forest for classification

Outcome
Achieved reliable face verification by combining classical image processing with machine learning, suitable for simple access control or attendance systems.

