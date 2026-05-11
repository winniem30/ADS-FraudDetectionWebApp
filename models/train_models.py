"""
Train all AML detection models:
  1. Random Forest        - primary classifier
  2. Logistic Regression  - probability baseline
  3. XGBoost              - gradient boosting
  4. Isolation Forest     - anomaly detection (unsupervised)
  5. One-Class SVM        - outlier detection (unsupervised)
"""

import os, pickle, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed. Skipping xgb_model. Run: pip install xgboost")

FEATURE_COLS = ['amount', 'hour', 'day_of_week', 'txn_frequency',
                'amount_deviation', 'large_amount', 'rapid_movement', 'circular_pattern']

MODELS_DIR = os.path.dirname(__file__)

def engineer_features(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    col_map = {
        'sender': 'sender_account', 'from': 'sender_account',
        'receiver': 'receiver_account', 'to': 'receiver_account',
        'amt': 'amount', 'transaction_amount': 'amount',
        'date': 'timestamp', 'time': 'timestamp',
        'type': 'transaction_type',
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    for col in ['sender_account', 'receiver_account']:
        if col not in df.columns:
            df[col] = 'unknown'
    if 'amount' not in df.columns:
        df['amount'] = np.random.exponential(1000, len(df))
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range('2024-01-01', periods=len(df), freq='5min')

    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)

    df['hour'] = df['timestamp'].dt.hour.fillna(12)
    df['day_of_week'] = df['timestamp'].dt.dayofweek.fillna(0)

    sender_counts = df.groupby('sender_account')['amount'].transform('count')
    df['txn_frequency'] = sender_counts

    sender_means = df.groupby('sender_account')['amount'].transform('mean')
    df['amount_deviation'] = (df['amount'] - sender_means).abs()

    df['large_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)

    df['time_diff'] = df.groupby('sender_account')['timestamp'].diff().dt.total_seconds().fillna(0)
    df['rapid_movement'] = (df['time_diff'] < 300).astype(int)

    df['txn_pair'] = df.apply(
        lambda r: tuple(sorted([str(r['sender_account']), str(r['receiver_account'])])), axis=1)
    pair_counts = df.groupby('txn_pair')['amount'].transform('count')
    df['circular_pattern'] = (pair_counts > 2).astype(int)

    return df

def make_synthetic_labels(df):
    """Rule-based labels for training when no ground truth exists."""
    score = (
        df['large_amount'] * 25 +
        df['rapid_movement'] * 20 +
        df['circular_pattern'] * 25 +
        (df['txn_frequency'] > 10).astype(int) * 15 +
        (df['amount_deviation'] > 5000).astype(int) * 15
    )
    return (score >= 40).astype(int)

def train(data_path=None):
    # Load or synthesise data
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        df_raw = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_excel(data_path)
    else:
        print("No data path given — generating synthetic dataset for demo training")
        n = 5000
        np.random.seed(42)
        df_raw = pd.DataFrame({
            'sender_account': np.random.choice([f'ACC{i:04d}' for i in range(200)], n),
            'receiver_account': np.random.choice([f'ACC{i:04d}' for i in range(200)], n),
            'amount': np.concatenate([
                np.random.exponential(500, int(n * 0.85)),
                np.random.exponential(50000, int(n * 0.15))
            ])[:n],
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='3min'),
        })

    df = engineer_features(df_raw)

    label_col = next((c for c in df.columns if c in ['is_fraud', 'label', 'fraud', 'suspicious', 'flagged']), None)
    if label_col:
        print(f"Using ground-truth labels from column: '{label_col}'")
        y = df[label_col].astype(int)
    else:
        print("No label column found — using rule-based synthetic labels")
        y = make_synthetic_labels(df)

    X = df[FEATURE_COLS].fillna(0)
    print(f"Dataset: {len(X)} rows | Fraud rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    results = {}

    # 1. Random Forest
    print("\n[1/5] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred))
    try: results['rf_auc'] = round(roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]), 4)
    except: pass
    with open(os.path.join(MODELS_DIR, 'rf_model.pkl'), 'wb') as f:
        pickle.dump(rf, f)

    # 2. Logistic Regression
    print("\n[2/5] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train_sc, y_train)
    y_pred = lr.predict(X_test_sc)
    print(classification_report(y_test, y_pred))
    try: results['lr_auc'] = round(roc_auc_score(y_test, lr.predict_proba(X_test_sc)[:, 1]), 4)
    except: pass
    with open(os.path.join(MODELS_DIR, 'lr_model.pkl'), 'wb') as f:
        pickle.dump(lr, f)

    # 3. XGBoost
    if HAS_XGB:
        print("\n[3/5] Training XGBoost...")
        scale_pos = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                             scale_pos_weight=scale_pos, use_label_encoder=False,
                             eval_metric='logloss', random_state=42)
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        print(classification_report(y_test, y_pred))
        try: results['xgb_auc'] = round(roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1]), 4)
        except: pass
        with open(os.path.join(MODELS_DIR, 'xgb_model.pkl'), 'wb') as f:
            pickle.dump(xgb, f)
    else:
        print("\n[3/5] Skipping XGBoost (not installed)")

    # 4. Isolation Forest (unsupervised anomaly detection)
    print("\n[4/5] Training Isolation Forest...")
    iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    # Convert: -1 (anomaly) → 1 (fraud), 1 (normal) → 0
    iso_pred = (iso.predict(X_test) == -1).astype(int)
    print(classification_report(y_test, iso_pred))
    with open(os.path.join(MODELS_DIR, 'iso_model.pkl'), 'wb') as f:
        pickle.dump(iso, f)

    # 5. One-Class SVM (trained on normal transactions only)
    print("\n[5/5] Training One-Class SVM...")
    X_normal = X_train_sc[y_train == 0]
    ocsvm = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
    ocsvm.fit(X_normal)
    ocsvm_pred = (ocsvm.predict(X_test_sc) == -1).astype(int)
    print(classification_report(y_test, ocsvm_pred))
    with open(os.path.join(MODELS_DIR, 'svm_model.pkl'), 'wb') as f:
        pickle.dump(ocsvm, f)

    # Save accuracy summary
    acc_path = os.path.join(MODELS_DIR, 'accuracy.txt')
    with open(acc_path, 'w') as f:
        f.write("=== AML Model Training Results ===\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    print(f"\nAll models saved to {MODELS_DIR}")
    print("Accuracy summary:", results)

if __name__ == '__main__':
    import sys
    data = sys.argv[1] if len(sys.argv) > 1 else None
    train(data)
