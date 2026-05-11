# AML Detection System — ADS (Accurate Detection Services)

AI-Powered Money Laundering Detection using Machine Learning and Network Analysis.
**Cyber Hackathon 2025 Finalist** · Patna, Bihar

---

## Features

| Feature | Details |
|---|---|
| ML Models | Random Forest · Logistic Regression · XGBoost · Isolation Forest · One-Class SVM |
| Risk Scoring | Per-transaction Low / Medium / High risk with 0–100 score |
| SpyderMap | Interactive D3.js transaction network graph |
| Feature Engineering | Transaction frequency · Amount deviation · Rapid movement · Circular patterns |
| Export | CSV · Excel (multi-sheet) · PDF report |
| Upload | CSV / XLSX / XLS · Up to 50MB |

---

## Setup

```bash
cd money_laundering_app
pip install -r requirements.txt
```

### (Optional) Train models on your own data
```bash
python models/train_models.py path/to/your_data.csv
```
If no data path is given, it trains on a synthetic dataset automatically.

### Run the app
```bash
python app.py
```
Open: http://127.0.0.1:5000

---

## Expected CSV Columns

| Column | Aliases Accepted |
|---|---|
| `sender_account` | sender, from |
| `receiver_account` | receiver, to |
| `amount` | amt, transaction_amount |
| `timestamp` | date, time, datetime |

Any CSV with transaction data will work — columns are auto-mapped.

---

## Project Architecture

```
money_laundering_app/
├── app.py                  ← Flask backend (all routes)
├── requirements.txt
├── models/
│   ├── train_models.py     ← Train all 5 ML models
│   ├── rf_model.pkl        ← Random Forest (generated after training)
│   ├── lr_model.pkl        ← Logistic Regression
│   ├── xgb_model.pkl       ← XGBoost
│   ├── iso_model.pkl       ← Isolation Forest
│   ├── svm_model.pkl       ← One-Class SVM
│   └── accuracy.txt        ← Training results
├── templates/
│   ├── upload.html         ← File upload page
│   └── dashboard.html      ← AML Dashboard + SpyderMap
├── static/
├── data/
└── uploads/                ← Uploaded files stored here
```

---

## Tech Stack

- **Backend**: Python, Flask
- **ML**: scikit-learn, XGBoost
- **Network Graph**: D3.js (SpyderMap)
- **Export**: reportlab (PDF), openpyxl (Excel)
- **Frontend**: HTML, CSS, JavaScript

---

## Team
Tejaswini Medandrao — Tech Lead  
Stanley College of Engineering and Technology | Class of 2027
