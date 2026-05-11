from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import pandas as pd
import numpy as np
import os, json, pickle
from werkzeug.utils import secure_filename
from datetime import datetime
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    models = {}
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    for name in ['rf_model', 'lr_model', 'xgb_model', 'iso_model', 'svm_model']:
        path = os.path.join(model_dir, f'{name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
    return models

def preprocess(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Map common column names
    col_map = {
        'sender': 'sender_account', 'from': 'sender_account',
        'receiver': 'receiver_account', 'to': 'receiver_account',
        'amt': 'amount', 'transaction_amount': 'amount',
        'date': 'timestamp', 'time': 'timestamp', 'datetime': 'timestamp',
        'type': 'transaction_type', 'txn_type': 'transaction_type',
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # Ensure required columns
    for col in ['sender_account', 'receiver_account', 'amount', 'timestamp']:
        if col not in df.columns:
            df[col] = 'unknown' if col != 'amount' else 0.0

    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    except:
        df['timestamp'] = pd.Timestamp.now()

    df = df.sort_values('timestamp').reset_index(drop=True)

    # Feature engineering
    df['hour'] = df['timestamp'].dt.hour.fillna(12)
    df['day_of_week'] = df['timestamp'].dt.dayofweek.fillna(0)

    sender_counts = df.groupby('sender_account')['amount'].transform('count')
    df['txn_frequency'] = sender_counts

    sender_means = df.groupby('sender_account')['amount'].transform('mean')
    df['amount_deviation'] = (df['amount'] - sender_means).abs()

    df['large_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)

    df['time_diff'] = df.groupby('sender_account')['timestamp'].diff().dt.total_seconds().fillna(0)
    df['rapid_movement'] = (df['time_diff'] < 300).astype(int)

    # Circular transaction detection
    df['txn_pair'] = df.apply(lambda r: tuple(sorted([str(r['sender_account']), str(r['receiver_account'])])), axis=1)
    pair_counts = df.groupby('txn_pair')['amount'].transform('count')
    df['circular_pattern'] = (pair_counts > 2).astype(int)

    return df

def compute_risk_score(row, pred_proba=None):
    score = 0
    if row.get('large_amount', 0): score += 25
    if row.get('rapid_movement', 0): score += 20
    if row.get('circular_pattern', 0): score += 25
    if row.get('txn_frequency', 1) > 10: score += 15
    if row.get('amount_deviation', 0) > 5000: score += 15
    if pred_proba is not None:
        score = int(score * 0.4 + pred_proba * 100 * 0.6)
    return min(score, 100)

def risk_label(score):
    if score >= 70: return 'High'
    if score >= 40: return 'Medium'
    return 'Low'

def analyze_df(df):
    df = preprocess(df)
    models = load_models()

    feature_cols = ['amount', 'hour', 'day_of_week', 'txn_frequency',
                    'amount_deviation', 'large_amount', 'rapid_movement', 'circular_pattern']
    X = df[feature_cols].fillna(0)

    results = pd.DataFrame()
    results['sender_account'] = df.get('sender_account', 'unknown')
    results['receiver_account'] = df.get('receiver_account', 'unknown')
    results['amount'] = df['amount']
    results['timestamp'] = df['timestamp'].astype(str)
    results['txn_frequency'] = df['txn_frequency']
    results['amount_deviation'] = df['amount_deviation'].round(2)
    results['rapid_movement'] = df['rapid_movement']
    results['circular_pattern'] = df['circular_pattern']
    results['large_amount'] = df['large_amount']

    # Model predictions
    pred_proba = np.zeros(len(X))
    if 'rf_model' in models:
        try:
            pred_proba = models['rf_model'].predict_proba(X)[:, 1]
        except: pass

    results['risk_score'] = [
        compute_risk_score(row, pred_proba[i])
        for i, (_, row) in enumerate(df[feature_cols].iterrows())
    ]
    results['risk_level'] = results['risk_score'].apply(risk_label)
    results['flagged'] = (results['risk_level'] == 'High').astype(int)

    # Model votes
    votes = []
    for i in range(len(X)):
        row_votes = {}
        for mname, model in models.items():
            try:
                p = model.predict(X.iloc[[i]])[0]
                row_votes[mname] = int(p)
            except: pass
        votes.append(json.dumps(row_votes))
    results['model_votes'] = votes

    return results

# ── Routes ──────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return render_template('upload.html', error='Please upload a valid CSV or Excel file.')

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
        results = analyze_df(df)
        results.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv'), index=False)

        summary = {
            'total': len(results),
            'high': int((results['risk_level'] == 'High').sum()),
            'medium': int((results['risk_level'] == 'Medium').sum()),
            'low': int((results['risk_level'] == 'Low').sum()),
            'flagged': int(results['flagged'].sum()),
            'filename': filename,
        }
        return render_template('dashboard.html', summary=summary,
                               records=results.head(500).to_dict('records'))
    except Exception as e:
        return render_template('upload.html', error=f'Error processing file: {str(e)}')

@app.route('/api/results')
def api_results():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
    if not os.path.exists(path):
        return jsonify({'error': 'No results yet'})
    df = pd.read_csv(path)
    return jsonify(df.to_dict('records'))

@app.route('/api/network')
def api_network():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
    if not os.path.exists(path):
        return jsonify({'nodes': [], 'links': []})
    df = pd.read_csv(path)
    df = df.dropna(subset=['sender_account', 'receiver_account'])

    nodes_set = set(df['sender_account'].astype(str)) | set(df['receiver_account'].astype(str))
    risk_map = {}
    for _, row in df.iterrows():
        s = str(row['sender_account'])
        risk_map[s] = max(risk_map.get(s, 0), row.get('risk_score', 0))

    nodes = [{'id': n, 'risk': risk_map.get(n, 0),
               'level': risk_label(risk_map.get(n, 0))} for n in nodes_set]

    links = []
    for _, row in df.iterrows():
        links.append({
            'source': str(row['sender_account']),
            'target': str(row['receiver_account']),
            'amount': float(row.get('amount', 0)),
            'risk': row.get('risk_score', 0),
        })

    # Limit for performance
    return jsonify({'nodes': nodes[:200], 'links': links[:500]})

@app.route('/export/csv')
def export_csv():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
    if not os.path.exists(path):
        return redirect(url_for('index'))
    return send_file(path, as_attachment=True, download_name='aml_report.csv')

@app.route('/export/excel')
def export_excel():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
    if not os.path.exists(path):
        return redirect(url_for('index'))
    df = pd.read_csv(path)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='AML Analysis')
        high = df[df['risk_level'] == 'High']
        if not high.empty:
            high.to_excel(writer, index=False, sheet_name='High Risk')
    out.seek(0)
    return send_file(out, as_attachment=True,
                     download_name='aml_report.xlsx',
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.route('/export/pdf')
def export_pdf():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
    if not os.path.exists(path):
        return redirect(url_for('index'))
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet

        df = pd.read_csv(path)
        high = df[df['risk_level'] == 'High']

        out = io.BytesIO()
        doc = SimpleDocTemplate(out, pagesize=landscape(A4))
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph('AML Fraud Detection Report', styles['Title']))
        elements.append(Paragraph(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', styles['Normal']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f'Total Transactions: {len(df)} | High Risk: {len(high)} | Flagged: {int(df["flagged"].sum())}', styles['Normal']))
        elements.append(Spacer(1, 12))

        cols = ['sender_account', 'receiver_account', 'amount', 'risk_score', 'risk_level', 'timestamp']
        display = high[cols].head(100) if not high.empty else df[cols].head(100)
        data = [cols] + display.values.tolist()
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTSIZE', (0,0), (-1,-1), 7),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ]))
        elements.append(t)
        doc.build(elements)
        out.seek(0)
        return send_file(out, as_attachment=True, download_name='aml_report.pdf',
                         mimetype='application/pdf')
    except ImportError:
        return 'reportlab not installed. Run: pip install reportlab', 500

if __name__ == '__main__':
    app.run(debug=True)
