ADS(Accurate Detection Services) Fraud Detection System
Cyber Hackathon 2025 Finalist · Patna, Bihar
Overview

The ADS Fraud Detection System is a machine learning-based cybersecurity solution developed to detect fraudulent activities in digital advertising datasets.

This project was built and presented at the Cyber Hackathon 2025 (Patna, Bihar), where it was selected as a Finalist for its practical implementation and real-world relevance.

The system enables users to upload structured datasets and performs fraud detection using supervised learning models, producing interpretable and actionable outputs.

Problem Context

Digital advertising ecosystems are increasingly affected by:

Invalid click generation (bot traffic)
Impression manipulation
Automated interaction patterns
Traffic spoofing

These issues lead to financial loss, unreliable analytics, and reduced trust in ad platforms.

Solution Approach

The system follows a structured pipeline:

CSV Upload → Preprocessing → Feature Engineering → Model Inference → Output → Visualization

It combines data preprocessing, feature engineering, and machine learning classification to identify fraudulent behavior patterns efficiently.

Core Components
Data Ingestion
Accepts CSV files containing ad interaction logs
Supports attributes such as IP address, timestamp, device, and session data
Data Processing
Missing value handling
Encoding of categorical variables
Feature scaling
Feature Engineering
Click frequency per IP
Time gap between clicks
Session-level activity metrics
Machine Learning Models
Model	Purpose	Strength
Random Forest	Primary classification model	Handles complex patterns, high accuracy
Logistic Regression	Secondary validation model	Interpretable, probability-based output
Output

The system generates:

Fraud / Legitimate classification
Probability scores
Flagged suspicious records

These outputs can be used for further investigation or integration into monitoring systems.

Visualization Layer

The system is designed to support advanced visualization:

Statistical plots using Python libraries
Planned integration with Tableau for:
Network-based spider maps
Fraud cluster identification
Pattern tracking across entities
Key Features
Structured CSV-based workflow
Dual-model fraud detection (Random Forest + Logistic Regression)
Clean and interpretable outputs
Modular and scalable design
Visualization-ready pipeline
Technology Stack

Backend

Python (Flask)

Machine Learning

Scikit-learn
Pandas, NumPy

Frontend

HTML, CSS, JavaScript

Visualization

Matplotlib, Seaborn
Tableau (integration-ready)

Database

SQLite / MySQL
Project Structure
ad-fraud-detection/
│── app.py
│── models/
│── static/
│── templates/
│── dataset/
│── requirements.txt
Setup and Execution
git clone https://github.com/your-username/ad-fraud-detection.git
cd ad-fraud-detection
pip install -r requirements.txt
python app.py

Access the application at:

http://127.0.0.1:5000/
Achievements
Finalist – Cyber Hackathon 2025, Patna
Recognized for applied cybersecurity innovation
Demonstrates real-world applicability in AdTech systems
Future Enhancements
Real-time data streaming integration
Advanced models (XGBoost, Deep Learning)
Automated fraud prevention mechanisms
Full Tableau dashboard integration
API-based deployment for external systems
Team

Tejaswini M – Team Lead
[Add Team Members]

Closing Note

This project reflects the application of machine learning in solving a critical cybersecurity challenge. It establishes a foundation for building fraud-resilient, data-driven advertising systems with scalable architecture and extensible analytics capabilities.
