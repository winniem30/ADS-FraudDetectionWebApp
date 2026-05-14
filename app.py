import os, pickle, io, logging
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = ['amount', 'hour', 'day_of_week', 'txn_frequency',
                'amount_deviation', 'large_amount', 'rapid_movement', 'circular_pattern']
COLUMN_ALIASES = {
    'sender': 'sender_account', 'from': 'sender_account', 'payer': 'sender_account', 'source': 'sender_account',
    'receiver': 'receiver_account', 'to': 'receiver_account', 'payee': 'receiver_account', 'destination': 'receiver_account',
    'amt': 'amount', 'transaction_amount': 'amount', 'value': 'amount',
    'date': 'timestamp', 'time': 'timestamp', 'datetime': 'timestamp', 'created_at': 'timestamp', 'txn_date': 'timestamp',
    'sender_location': 'location', 'ip_address': 'ip_address', 'device_id': 'device_id', 'status': 'status',
    'sender_created_at': 'sender_created_at',
}

def _save(name, obj):
    with open(os.path.join(MODELS_DIR, f'{name}.pkl'), 'wb') as f:
        pickle.dump(obj, f)


def train_all_models():
    """Bootstrap compact models for fresh deployments with empty models/."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM

    np.random.seed(42)
    n = 5000
    accounts = [f'ACC{i:04d}' for i in range(200)]
    df_raw = pd.DataFrame({
        'sender_account': np.random.choice(accounts, n),
        'receiver_account': np.random.choice(accounts, n),
        'amount': np.where(np.random.rand(n) < 0.1, np.random.exponential(70000, n), np.random.exponential(1000, n)),
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='4min')
    })
    df = _engineer(df_raw)
    X = df[FEATURE_COLS].fillna(0)
    y = ((df['large_amount'] == 1) | (df['rapid_movement'] == 1) | (df['circular_pattern'] == 1)).astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X_tr, y_tr)
    _save('rf_model', rf)

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    lr = LogisticRegression(max_iter=800, class_weight='balanced', random_state=42)
    lr.fit(X_tr_sc, y_tr)
    _save('lr_model', (lr, sc))

    ocsvm = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
    ocsvm.fit(X_tr_sc[y_tr == 0] if (y_tr == 0).any() else X_tr_sc)
    _save('svm_model', (ocsvm, sc))
    log.info("Model bootstrap complete.")

def _engineer(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    for old, new in COLUMN_ALIASES.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    for col in ['sender_account', 'receiver_account', 'location', 'status', 'device_id', 'ip_address']:
        if col not in df.columns:
            df[col] = 'UNKNOWN'
    if 'amount' not in df.columns:
        df['amount'] = np.random.exponential(500, len(df))
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range('2024-01-01', periods=len(df), freq='5min')
    if 'sender_created_at' not in df.columns:
        df['sender_created_at'] = df['timestamp']

    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['timestamp'] = df['timestamp'].ffill().fillna(pd.Timestamp('2024-01-01'))
    df['sender_created_at'] = pd.to_datetime(df['sender_created_at'], errors='coerce').fillna(df['timestamp'])

    df = df.sort_values('timestamp').reset_index(drop=True)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['txn_frequency'] = df.groupby('sender_account')['amount'].transform('count')
    sender_mean = df.groupby('sender_account')['amount'].transform('mean').replace(0, 1)
    df['amount_deviation'] = (df['amount'] - sender_mean).abs()
    df['large_amount'] = (df['amount'] > max(100000, df['amount'].quantile(0.95))).astype(int)
    df['time_diff'] = df.groupby('sender_account')['timestamp'].diff().dt.total_seconds().fillna(999999)
    df['rapid_movement'] = (df['time_diff'] < 300).astype(int)
    df['txn_pair'] = df['sender_account'].astype(str) + '->' + df['receiver_account'].astype(str)
    df['circular_pattern'] = df.duplicated(['sender_account', 'receiver_account'], keep=False).astype(int)
    df['many_receivers'] = (df.groupby('sender_account')['receiver_account'].transform('nunique') > 5).astype(int)
    df['new_account'] = ((df['timestamp'] - df['sender_created_at']).dt.total_seconds() < 7 * 24 * 3600).astype(int)
    return df

# keep original training/inference loading from existing models

def _load(name):
    path = os.path.join(MODELS_DIR, f'{name}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def _risk_label(score):
    return 'High' if score >= 71 else ('Medium' if score >= 31 else 'Low')

def _rule_reasons(row):
    reasons, rule_score = [], 0
    if 0 <= row['hour'] < 5: reasons.append(f"Odd Timing ({int(row['hour'])}:xx AM)"); rule_score += 10
    if row['large_amount'] == 1: reasons.append('High Amount (> ₹1,00,000)'); rule_score += 18
    if row['rapid_movement'] == 1: reasons.append('Rapid repeated transfers'); rule_score += 12
    if row['many_receivers'] == 1: reasons.append('Sender routing to many receivers'); rule_score += 10
    if row['circular_pattern'] == 1: reasons.append('Repeated / circular routing path'); rule_score += 12
    if row['amount_deviation'] > max(25000, row['amount'] * 0.6): reasons.append('Sudden spike vs sender baseline'); rule_score += 10
    if row['is_weekend'] == 1: reasons.append('Weekend transaction pattern'); rule_score += 6
    if row['new_account'] == 1: reasons.append('Newly created account activity'); rule_score += 12
    if str(row.get('status', '')).lower() in {'failed', 'reversed'}: reasons.append('Failed/Reversed transaction'); rule_score += 10
    if round(float(row['amount']), -3) == float(row['amount']) and row['amount'] > 10000: reasons.append('Repeated round-number transfer pattern'); rule_score += 5
    return min(rule_score, 100), reasons

def analyze_df(raw_df):
    df = _engineer(raw_df)
    X = df[FEATURE_COLS].fillna(0).astype(np.float32)
    n = len(X)
    if n == 0:
        cols = ['sender_account', 'receiver_account', 'amount', 'timestamp', 'location', 'rf_score', 'lr_score', 'xgb_score',
                'svm_flag', 'risk_score', 'risk_level', 'rapid_movement', 'circular_pattern', 'large_amount',
                'txn_frequency', 'amount_deviation', 'fraud_reasons', 'flagged']
        return pd.DataFrame(columns=cols)
    if n > 30000:
        df = df.head(30000).copy(); X = X.head(30000).copy(); n = len(X)

    rf = _load('rf_model'); lr_bundle = _load('lr_model'); xgb = _load('xgb_model'); svm_bundle = _load('svm_model')
    if rf is None and lr_bundle is None and svm_bundle is None and xgb is None:
        log.warning("No persisted models found, bootstrapping default models.")
        train_all_models()
        rf = _load('rf_model'); lr_bundle = _load('lr_model'); xgb = _load('xgb_model'); svm_bundle = _load('svm_model')
    lr, lr_sc = (lr_bundle if isinstance(lr_bundle, tuple) else (lr_bundle, None))
    ocsvm, sv_sc = (svm_bundle if isinstance(svm_bundle, tuple) else (svm_bundle, None))
    X_sc = lr_sc.transform(X) if lr_sc is not None else X.values

    rf_proba = rf.predict_proba(X)[:, 1] if rf else np.zeros(n)
    lr_proba = lr.predict_proba(X_sc)[:, 1] if lr else np.zeros(n)
    xgb_proba = xgb.predict_proba(X)[:, 1] if xgb else np.zeros(n)
    svm_pred = (ocsvm.predict(sv_sc.transform(X) if sv_sc else X) == -1).astype(int) if ocsvm else np.zeros(n)

    prob_stack = [p for p in [rf_proba, lr_proba, xgb_proba] if np.any(p)]
    ensemble_prob = np.mean(prob_stack, axis=0) if prob_stack else np.zeros(n)

    rows = []
    for i, row in df.iterrows():
        rule_score, reasons = _rule_reasons(row)
        ml_prob = float(ensemble_prob[i] * 100)
        anomaly = int(svm_pred[i]) * 100
        risk = int(np.clip((ml_prob * 0.45) + (rule_score * 0.45) + (anomaly * 0.10), 0, 100))
        if ml_prob > 80:
            reasons.append(f"ML fraud probability: {ml_prob:.1f}%")
        if not reasons and risk > 30:
            reasons.append('Behavioral anomaly mix triggered risk threshold')
        rows.append({
            'sender_account': str(row['sender_account']), 'receiver_account': str(row['receiver_account']),
            'amount': round(float(row['amount']), 2), 'timestamp': str(row['timestamp']), 'location': str(row.get('location', 'UNKNOWN')),
            'rf_score': round(float(rf_proba[i] * 100), 1), 'lr_score': round(float(lr_proba[i] * 100), 1), 'xgb_score': round(float(xgb_proba[i] * 100), 1),
            'svm_flag': int(svm_pred[i]), 'risk_score': risk, 'risk_level': _risk_label(risk),
            'rapid_movement': int(row['rapid_movement']), 'circular_pattern': int(row['circular_pattern']), 'large_amount': int(row['large_amount']),
            'txn_frequency': int(row['txn_frequency']), 'amount_deviation': round(float(row['amount_deviation']), 2),
            'fraud_reasons': '; '.join(reasons[:5]),
        })
    cols = ['sender_account', 'receiver_account', 'amount', 'timestamp', 'location', 'rf_score', 'lr_score', 'xgb_score',
            'svm_flag', 'risk_score', 'risk_level', 'rapid_movement', 'circular_pattern', 'large_amount',
            'txn_frequency', 'amount_deviation', 'fraud_reasons']
    res = pd.DataFrame(rows, columns=cols)
    if res.empty:
        res['flagged'] = pd.Series(dtype=int)
        return res
    res['flagged'] = ((res['risk_level'] == 'High') | (res['risk_score'] >= 65)).astype(int)
    return res

# routes unchanged with enhanced summary

def allowed_file(fn): return '.' in fn and fn.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
_required = ['rf_model', 'lr_model', 'svm_model']
if not all(os.path.exists(os.path.join(MODELS_DIR, f'{m}.pkl')) for m in _required):
    log.info("Persisted models missing at startup. Training bootstrap models.")
    train_all_models()

@app.route('/')
def index(): return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files: return redirect(url_for('index'))
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return render_template('upload.html', error='Please upload a valid CSV or Excel file.')
    filename = secure_filename(file.filename); filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename); file.save(filepath)
    try:
        raw = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
        results = analyze_df(raw)
        results.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv'), index=False)
        flagged = results[results['flagged'] == 1]
        summary = {'total': len(results), 'high': int((results['risk_level'] == 'High').sum()), 'medium': int((results['risk_level'] == 'Medium').sum()), 'low': int((results['risk_level'] == 'Low').sum()), 'flagged': int(results['flagged'].sum()), 'filename': filename,
                   'top_locations': flagged['location'].value_counts().head(5).to_dict(),
                   'top_accounts': flagged['sender_account'].value_counts().head(5).to_dict()}
        return render_template('dashboard.html', summary=summary, records=results.head(500).to_dict('records'))
    except Exception as e:
        log.exception('Analysis failed'); return render_template('upload.html', error=f'Error: {str(e)}')

@app.route('/api/network')
def api_network():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
    if not os.path.exists(path): return jsonify({'nodes': [], 'links': []})
    df = pd.read_csv(path)
    df = df[df['flagged'] == 1].dropna(subset=['sender_account', 'receiver_account']).head(400)
    if df.empty: return jsonify({'nodes': [], 'links': []})
    nodes = []
    for acc, risk in pd.concat([df[['sender_account', 'risk_score']].rename(columns={'sender_account': 'id'}), df[['receiver_account', 'risk_score']].rename(columns={'receiver_account': 'id'})]).groupby('id')['risk_score'].max().items():
        nodes.append({'id': str(acc), 'risk': float(risk), 'level': _risk_label(risk)})
    links = [{'source': str(r['sender_account']), 'target': str(r['receiver_account']), 'amount': float(r['amount']), 'risk': float(r['risk_score']), 'reason': str(r.get('fraud_reasons', ''))} for _, r in df.iterrows()]
    return jsonify({'nodes': nodes[:250], 'links': links[:500]})

@app.route('/export/csv')
def export_csv():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
    if not os.path.exists(path): return redirect(url_for('index'))
    return send_file(path, as_attachment=True, download_name='aml_report.csv')


@app.route('/export/excel')
def export_excel():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
    if not os.path.exists(path): return redirect(url_for('index'))
    df = pd.read_csv(path)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, index=False, sheet_name='All Transactions')
        df[df['risk_level'] == 'High'].to_excel(w, index=False, sheet_name='High Risk')
        df[df['risk_level'] == 'Medium'].to_excel(w, index=False, sheet_name='Medium Risk')
    out.seek(0)
    return send_file(out, as_attachment=True, download_name='aml_report.xlsx',
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


@app.route('/export/pdf')
def export_pdf():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
    if not os.path.exists(path): return redirect(url_for('index'))
    try:
        from reportlab.lib.pagesizes import landscape, A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet

        df = pd.read_csv(path)
        high = df[df['risk_level'] == 'High']
        out = io.BytesIO()
        doc = SimpleDocTemplate(out, pagesize=landscape(A4))
        st = getSampleStyleSheet()
        els = [
            Paragraph('AML Fraud Detection Report — ADS', st['Title']),
            Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", st['Normal']),
            Spacer(1, 12),
            Paragraph(
                f'Total: {len(df)} | High Risk: {len(high)} | '
                f"Medium: {int((df['risk_level']=='Medium').sum())} | "
                f"Flagged: {int(df['flagged'].sum())}",
                st['Normal']),
            Spacer(1, 12),
        ]
        cols = ['sender_account', 'receiver_account', 'amount', 'risk_score', 'risk_level', 'rapid_movement', 'circular_pattern', 'timestamp']
        disp = high[cols].head(100) if not high.empty else df[cols].head(100)
        data = [cols] + [list(map(str, row)) for row in disp.values.tolist()]
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0a1628')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
        ]))
        els.append(t)
        doc.build(els)
        out.seek(0)
        return send_file(out, as_attachment=True, download_name='aml_report.pdf', mimetype='application/pdf')
    except ImportError:
        return 'reportlab not installed', 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
