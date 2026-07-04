"""
Training Script for Dataset 1 (Daily Transactions Dataset)
Trains Random Forest, SVM, and XGBoost models.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import joblib
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.dataset1_preprocessor import Dataset1Preprocessor


class Dataset1Trainer:
    """
    Trainer for Dataset 1 (Daily Transactions Dataset).
    """
    
    def __init__(self, data_path: str, models_dir: str = 'models'):
        self.data_path = data_path
        self.models_dir = models_dir
        self.preprocessor = Dataset1Preprocessor()
        self.models = {}
        self.metrics = {}
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset.
        
        Returns:
            DataFrame with the loaded data
        """
        if self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
        elif self.data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(self.data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        
        print(f"Loaded {len(df)} rows from {self.data_path}")
        return df
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create labels for training (synthetic labels for unsupervised dataset).
        
        Since Dataset 1 is a daily transactions dataset without fraud labels,
        we'll create synthetic labels based on amount thresholds and patterns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added 'label' column
        """
        df = df.copy()
        
        # Normalize column names
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Create synthetic labels based on amount and category
        # High amounts in certain categories are flagged as suspicious
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        
        # Define suspicious patterns
        suspicious_conditions = (
            (df['amount'] > df['amount'].quantile(0.95)) |  # Very high amounts
            (df['category'].isin(['Transfer', 'Investment', 'Loan']))  # Risky categories
        )
        
        df['label'] = suspicious_conditions.astype(int)
        
        print(f"Label distribution:\n{df['label'].value_counts()}")
        return df
    
    def train(self):
        """
        Train all models for Dataset 1.
        """
        print("=" * 60)
        print("Training Dataset 1 Models")
        print("=" * 60)
        
        # Load data
        df = self.load_data()
        
        # Create labels
        df = self.create_labels(df)
        
        # Preprocess data
        print("\nPreprocessing data...")
        X = self.preprocessor.fit_transform(df)
        y = df['label'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train Random Forest
        print("\n" + "=" * 60)
        print("Training Random Forest...")
        print("=" * 60)
        self.train_random_forest(X_train, X_test, y_train, y_test)
        
        # Train SVM
        print("\n" + "=" * 60)
        print("Training SVM...")
        print("=" * 60)
        self.train_svm(X_train, X_test, y_train, y_test)
        
        # Train XGBoost
        print("\n" + "=" * 60)
        print("Training XGBoost...")
        print("=" * 60)
        self.train_xgboost(X_train, X_test, y_train, y_test)
        
        # Save preprocessing pipeline
        print("\n" + "=" * 60)
        print("Saving preprocessing pipeline...")
        print("=" * 60)
        self.preprocessor.save_pipeline(self.models_dir)
        
        # Print summary
        self.print_summary()
        
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """
        Train Random Forest model.
        """
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        self.models['random_forest'] = rf
        self.metrics['random_forest'] = metrics
        
        # Save model
        joblib.dump(rf, os.path.join(self.models_dir, 'dataset1_rf.pkl'))
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print("Model saved as dataset1_rf.pkl")
        
    def train_svm(self, X_train, X_test, y_train, y_test):
        """
        Train SVM model.
        """
        svm = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        self.models['svm'] = svm
        self.metrics['svm'] = metrics
        
        # Save model
        joblib.dump(svm, os.path.join(self.models_dir, 'dataset1_svm.pkl'))
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print("Model saved as dataset1_svm.pkl")
        
    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """
        Train XGBoost model.
        """
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        self.models['xgboost'] = xgb_model
        self.metrics['xgboost'] = metrics
        
        # Save model
        joblib.dump(xgb_model, os.path.join(self.models_dir, 'dataset1_xgb.pkl'))
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print("Model saved as dataset1_xgb.pkl")
        
    def print_summary(self):
        """
        Print training summary.
        """
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        
        for model_name, metrics in self.metrics.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        
        # Save metrics
        joblib.dump(self.metrics, os.path.join(self.models_dir, 'dataset1_metrics.pkl'))
        print("\nMetrics saved as dataset1_metrics.pkl")


def main():
    """
    Main function to run training.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train models for Dataset 1')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset file')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory to save models')
    
    args = parser.parse_args()
    
    trainer = Dataset1Trainer(args.data, args.models_dir)
    trainer.train()


if __name__ == '__main__':
    main()
