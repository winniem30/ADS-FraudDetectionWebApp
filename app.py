import os, json, pickle, io, logging
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Feature columns used by all supervised models ──────────────────────────
FEATURE_COLS = ['amount', 'hour', 'day_of_week', 'txn_frequency',
                'amount_deviation', 'large_amount', 'rapid_movement', 'circular_pattern']

# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING  (runs automatically at startup if .pkl files are missing)
# ══════════════════════════════════════════════════════════════════════════════

def _engineer(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    aliases = {
        'sender': 'sender_account', 'from': 'sender_account',
        'payer': 'sender_account', 'source': 'sender_account',
        'receiver': 'receiver_account', 'to': 'receiver_account',
        'payee': 'receiver_account', 'destination': 'receiver_account',
        'amt': 'amount', 'transaction_amount': 'amount', 'value': 'amount',
        'date': 'timestamp', 'time': 'timestamp', 'datetime': 'timestamp',
        'created_at': 'timestamp', 'txn_date': 'timestamp',
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    for col in ['sender_account', 'receiver_account']:
        if col not in df.columns:
            df[col] = f'ACC_{np.random.randint(1000,9999)}'
    if 'amount' not in df.columns:
        df['amount'] = np.random.exponential(500, len(df))
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range('2024-01-01', periods=len(df), freq='5min')

    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)

    df['hour'] = df['timestamp'].dt.hour.fillna(12).astype(int)
    df['day_of_week'] = df['timestamp'].dt.dayofweek.fillna(0).astype(int)
    df['txn_frequency'] = df.groupby('sender_account')['amount'].transform('count')
    sender_mean = df.groupby('sender_account')['amount'].transform('mean')
    df['amount_deviation'] = (df['amount'] - sender_mean).abs()
    df['large_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
    df['time_diff'] = df.groupby('sender_account')['timestamp'].diff().dt.total_seconds().fillna(0)
    df['rapid_movement'] = (df['time_diff'] < 300).astype(int)
    df['txn_pair'] = df.apply(
        lambda r: tuple(sorted([str(r['sender_account']), str(r['receiver_account'])])), axis=1)
    pair_counts = df.groupby('txn_pair')['amount'].transform('count')
    df['circular_pattern'] = (pair_counts > 2).astype(int)
    return df


def _synthetic_labels(df):
    score = (
        df['large_amount'] * 25 +
        df['rapid_movement'] * 20 +
        df['circular_pattern'] * 25 +
        (df['txn_frequency'] > 10).astype(int) * 15 +
        (df['amount_deviation'] > df['amount_deviation'].quantile(0.8)).astype(int) * 15
    )
    return (score >= 40).astype(int)


def train_all_models():
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import OneClassSVM
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    log.info("Training all 5 AML models on synthetic data...")

    # Generate rich synthetic dataset
    np.random.seed(42)
    n = 8000
    accounts = [f'ACC{i:04d}' for i in range(300)]
    senders   = np.random.choice(accounts, n)
    receivers = np.random.choice(accounts, n)

    # Mix of normal + suspicious amounts
    amounts = np.where(
        np.random.rand(n) < 0.12,
        np.random.exponential(80000, n),   # large suspicious
        np.random.exponential(800, n)      # normal
    )
    timestamps = pd.date_range('2023-01-01', periods=n, freq='3min')

    df_raw = pd.DataFrame({
        'sender_account': senders,
        'receiver_account': receivers,
        'amount': amounts,
        'timestamp': timestamps,
    })

    df = _engineer(df_raw)
    X = df[FEATURE_COLS].fillna(0)
    y = _synthetic_labels(df)

    log.info(f"Dataset: {len(X)} rows | Fraud rate: {y.mean():.2%}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)

    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42,
                                 n_jobs=-1, class_weight='balanced')
    rf.fit(X_tr, y_tr)
    _save('rf_model', rf)
    log.info(f"RF accuracy: {rf.score(X_te, y_te):.3f}")

    # 2. Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_tr_sc, y_tr)
    # Wrap so it uses same (unscaled) input at inference — store scaler with it
    _save('lr_model', (lr, sc))
    log.info(f"LR accuracy: {lr.score(X_te_sc, y_te):.3f}")

    # 3. XGBoost
    try:
        from xgboost import XGBClassifier
        scale_pos = int((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                             scale_pos_weight=scale_pos, eval_metric='logloss',
                             random_state=42, use_label_encoder=False)
        xgb.fit(X_tr, y_tr)
        _save('xgb_model', xgb)
        log.info(f"XGB accuracy: {xgb.score(X_te, y_te):.3f}")
    except ImportError:
        log.warning("XGBoost not installed — skipping xgb_model")

    # 4. Isolation Forest  (unsupervised anomaly detection)
    iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=42, n_jobs=-1)
    iso.fit(X_tr)
    _save('iso_model', iso)
    log.info("Isolation Forest trained")

    # 5. One-Class SVM  (trained on normal transactions only)
    X_normal_sc = X_tr_sc[y_tr == 0]
    ocsvm = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
    ocsvm.fit(X_normal_sc)
    _save('svm_model', (ocsvm, sc))   # store scaler alongside
    log.info("One-Class SVM trained")

    log.info("All 5 models saved ✓")


def _save(name, obj):
    with open(os.path.join(MODELS_DIR, f'{name}.pkl'), 'wb') as f:
        pickle.dump(obj, f)


def _load(name):
    path = os.path.join(MODELS_DIR, f'{name}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


# Auto-train on first startup if models are missing
_required = ['rf_model', 'lr_model', 'iso_model', 'svm_model']
if not all(os.path.exists(os.path.join(MODELS_DIR, f'{m}.pkl')) for m in _required):
    train_all_models()


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _risk_label(score):
    if score >= 70: return 'High'
    if score >= 40: return 'Medium'
    return 'Low'


def _rule_score(row):
    s = 0
    if row.get('large_amount', 0):   s += 25
    if row.get('rapid_movement', 0): s += 20
    if row.get('circular_pattern', 0): s += 25
    if row.get('txn_frequency', 1) > 10: s += 15
    if row.get('amount_deviation', 0) > 5000: s += 15
    return s


def analyze_df(raw_df):
    df = _engineer(raw_df)
    X  = df[FEATURE_COLS].fillna(0)

    # ── Load all models ──────────────────────────────────────────────────────
    rf          = _load('rf_model')
    lr_bundle   = _load('lr_model')   # (lr, scaler) or just lr
    xgb         = _load('xgb_model')
    iso         = _load('iso_model')
    svm_bundle  = _load('svm_model')  # (ocsvm, scaler)

    # Unpack bundles
    lr, lr_sc   = (lr_bundle if isinstance(lr_bundle, tuple) else (lr_bundle, None))
    ocsvm, sv_sc = (svm_bundle if isinstance(svm_bundle, tuple) else (svm_bundle, None))

    X_sc = lr_sc.transform(X) if lr_sc is not None else X.values

    # ── Per-row predictions ──────────────────────────────────────────────────
    n = len(X)
    rf_proba  = rf.predict_proba(X)[:, 1]       if rf    else np.zeros(n)
    lr_proba  = lr.predict_proba(X_sc)[:, 1]    if lr    else np.zeros(n)
    xgb_proba = xgb.predict_proba(X)[:, 1]      if xgb   else np.zeros(n)
    iso_pred  = (iso.predict(X) == -1).astype(int) if iso else np.zeros(n)
    svm_pred  = (ocsvm.predict(sv_sc.transform(X) if sv_sc else X) == -1).astype(int) \
                    if ocsvm else np.zeros(n)

    # Ensemble: average of available probability models + anomaly detectors
    prob_stack = [p for p in [rf_proba, lr_proba, xgb_proba] if p.any()]
    anom_stack = [p for p in [iso_pred, svm_pred] if p.any()]
    ensemble_prob = np.mean(prob_stack, axis=0) if prob_stack else np.zeros(n)
    ensemble_anom = np.mean(anom_stack, axis=0) if anom_stack else np.zeros(n)

    # Final risk score = 50% rule-based + 35% ensemble ML prob + 15% anomaly
    rule_scores = np.array([
        _rule_score(dict(zip(FEATURE_COLS, X.iloc[i]))) for i in range(n)
    ])
    risk_scores = np.clip(
        rule_scores * 0.50 + ensemble_prob * 100 * 0.35 + ensemble_anom * 100 * 0.15,
        0, 100
    ).astype(int)

    # ── Build results dataframe ──────────────────────────────────────────────
    res = pd.DataFrame({
        'sender_account':   df['sender_account'].astype(str),
        'receiver_account': df['receiver_account'].astype(str),
        'amount':           df['amount'].round(2),
        'timestamp':        df['timestamp'].astype(str),
        'txn_frequency':    df['txn_frequency'].astype(int),
        'amount_deviation': df['amount_deviation'].round(2),
        'rapid_movement':   df['rapid_movement'].astype(int),
        'circular_pattern': df['circular_pattern'].astype(int),
        'large_amount':     df['large_amount'].astype(int),
        'rf_score':         (rf_proba * 100).round(1),
        'lr_score':         (lr_proba * 100).round(1),
        'xgb_score':        (xgb_proba * 100).round(1),
        'iso_flag':         iso_pred,
        'svm_flag':         svm_pred,
        'risk_score':       risk_scores,
    })
    res['risk_level'] = res['risk_score'].apply(_risk_label)
    res['flagged']    = (res['risk_level'] == 'High').astype(int)

    return res


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

def allowed_file(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
    file.save(filepath)

    try:
        raw = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
        results = analyze_df(raw)
        results.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv'), index=False)

        summary = {
            'total':    len(results),
            'high':     int((results['risk_level'] == 'High').sum()),
            'medium':   int((results['risk_level'] == 'Medium').sum()),
            'low':      int((results['risk_level'] == 'Low').sum()),
            'flagged':  int(results['flagged'].sum()),
            'filename': filename,
        }
        return render_template('dashboard.html',
                               summary=summary,
                               records=results.head(500).to_dict('records'))
    except Exception as e:
        log.exception("Analysis failed")
        return render_template('upload.html', error=f'Error: {str(e)}')


@app.route('/api/network')
def api_network():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
    if not os.path.exists(path):
        return jsonify({'nodes': [], 'links': []})

    df = pd.read_csv(path).dropna(subset=['sender_account', 'receiver_account'])
    risk_map = {}
    for _, row in df.iterrows():
        s = str(row['sender_account'])
        risk_map[s] = max(risk_map.get(s, 0), float(row.get('risk_score', 0)))

    nodes_set = set(df['sender_account'].astype(str)) | set(df['receiver_account'].astype(str))
    nodes = [{'id': n, 'risk': risk_map.get(n, 0), 'level': _risk_label(risk_map.get(n, 0))}
             for n in nodes_set]
    links = [{'source': str(r['sender_account']), 'target': str(r['receiver_account']),
              'amount': float(r.get('amount', 0)), 'risk': float(r.get('risk_score', 0))}
             for _, r in df.iterrows()]

    return jsonify({'nodes': nodes[:300], 'links': links[:600]})


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

        df   = pd.read_csv(path)
        high = df[df['risk_level'] == 'High']
        out  = io.BytesIO()
        doc  = SimpleDocTemplate(out, pagesize=landscape(A4))
        st   = getSampleStyleSheet()
        els  = [
            Paragraph('AML Fraud Detection Report — ADS', st['Title']),
            Paragraph(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', st['Normal']),
            Spacer(1, 12),
            Paragraph(
                f'Total: {len(df)} | High Risk: {len(high)} | '
                f'Medium: {int((df["risk_level"]=="Medium").sum())} | '
                f'Flagged: {int(df["flagged"].sum())}',
                st['Normal']),
            Spacer(1, 12),
        ]
        cols = ['sender_account', 'receiver_account', 'amount',
                'risk_score', 'risk_level', 'rapid_movement', 'circular_pattern', 'timestamp']
        disp = high[cols].head(100) if not high.empty else df[cols].head(100)
        data = [cols] + [list(map(str, row)) for row in disp.values.tolist()]
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0a1628')),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('FONTSIZE',   (0, 0), (-1, -1), 7),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, colors.HexColor('#f0f4f8')]),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
        ]))
        els.append(t)
        doc.build(els)
        out.seek(0)
        return send_file(out, as_attachment=True, download_name='aml_report.pdf',
                         mimetype='application/pdf')
    except ImportError:
        return 'reportlab not installed', 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
