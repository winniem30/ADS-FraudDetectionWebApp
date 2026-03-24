🚨 Ad Fraud Detection System
🏆 Finalist – Cyber Hackathon 2025, Patna, Bihar
📌 Abstract

The Ad Fraud Detection System is a machine learning-driven cybersecurity solution designed to identify and mitigate fraudulent activities in digital advertising ecosystems.

Developed as part of the Cyber Hackathon 2025 (Patna, Bihar), the project was recognized as a Finalist for its robust and scalable approach to fraud detection using structured data analytics and supervised learning models.

The system enables users to upload advertising datasets in CSV format, processes the data through trained models, and generates actionable insights by classifying interactions as fraudulent or legitimate. Additionally, it is designed to support advanced visualization through Tableau-based spider maps for network-level fraud analysis.

🎯 Problem Definition

Digital advertising platforms are increasingly vulnerable to sophisticated fraud mechanisms such as:

Click fraud and bot-driven traffic
Impression manipulation
Traffic spoofing and automated interactions

These activities result in significant financial losses, skewed analytics, and reduced platform trustworthiness. Detecting such anomalies in large-scale datasets remains a critical challenge.

💡 Proposed Solution

This project presents a data-driven fraud detection framework that integrates:

Supervised machine learning models for classification
Structured preprocessing and feature engineering
Analytical outputs with visualization support

The system is designed to be modular, scalable, and adaptable to real-world advertising data pipelines.

🧠 Methodology
1. Data Ingestion
Input: CSV datasets containing ad interaction logs
Attributes include IP address, timestamp, device information, and user behavior metrics
2. Data Preprocessing
Handling missing and inconsistent values
Encoding categorical features
Feature scaling and normalization
3. Feature Engineering

Key derived features include:

Click frequency per IP
Time interval between consecutive clicks
Session-based activity patterns
Behavioral consistency metrics
🤖 Machine Learning Models
🔹 Random Forest Classifier
Primary model for fraud detection
Captures complex, non-linear relationships
Provides high accuracy and robustness against noise
🔹 Logistic Regression
Secondary model for binary classification
Generates probability-based outputs
Used for validation and comparative analysis
⚙️ System Architecture

Pipeline Overview:

CSV Upload → Data Preprocessing → Feature Engineering →
Model Inference (RF + LR) → Fraud Classification → Visualization

📊 Output

The system generates:

Binary classification (Fraud / Legitimate)
Probability scores for each instance
Flagged suspicious records for further analysis
🕸️ Visualization (Planned Enhancement)

To enhance interpretability and investigation capabilities, the system is designed to integrate with Tableau for advanced visualization:

Spider (network) maps to represent relationships between entities
Identification of clustered fraudulent activities
Detection of coordinated attack patterns
⚙️ Key Features
📂 CSV-based data ingestion
🤖 Dual-model fraud detection (Random Forest + Logistic Regression)
📊 Analytical output generation
🔍 Scalable and modular design
🕸️ Visualization-ready architecture
🛠️ Technology Stack

Frontend:

HTML, CSS, JavaScript

Backend:

Python (Flask)

Machine Learning:

Scikit-learn
Pandas, NumPy

Visualization:

Matplotlib, Seaborn
Tableau (planned integration)

Database:

SQLite / MySQL
