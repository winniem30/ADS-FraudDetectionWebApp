from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load models
lr_model = joblib.load('models/lr_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')

@app.route('/')
def upload():
    return render_template('upload.html')

@app.route('/dashboard', methods=['POST'])
def dashboard():
    file = request.files['file']
    df = pd.read_csv(file)

    # Feature engineering
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 1. Suspicious location flag
    suspicious_locs = ['Dubai', 'London', 'Panama', 'Singapore']
    df['location_flag'] = df['location'].apply(lambda x: 1 if x in suspicious_locs else 0)

    # 2. Odd transaction hour (0–4 AM)
    df['odd_time'] = df['timestamp'].dt.hour.apply(lambda x: 1 if 0 <= x <= 4 else 0)

    # 3. Large amount flag (> 100000)
    df['large_amount'] = df['amount'].apply(lambda x: 1 if x > 100000 else 0)

    # Features for prediction
    features = df[['location_flag', 'odd_time', 'large_amount', 'business_account', 'amount']]

    # Make predictions
    df['LR_Prediction'] = lr_model.predict(features)
    df['RF_Prediction'] = rf_model.predict(features)

    # Reasoning logic
    def reason(row):
        reasons = []
        if row['location_flag']: reasons.append('Suspicious Location')
        if row['odd_time']: reasons.append('Odd Time')
        if row['large_amount']: reasons.append('Large Amount')
        if row['business_account']: reasons.append('Business Account')
        return ', '.join(reasons)

    df['Reason'] = df.apply(reason, axis=1)

    # Load model accuracy from file
    with open('models/accuracy.txt') as f:
        accuracy_data = f.read().splitlines()

    return render_template('dashboard.html', tables=df.to_dict('records'), accuracy=accuracy_data)

if __name__ == '__main__':
    app.run(debug=True)