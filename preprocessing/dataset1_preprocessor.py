"""
Dataset 1 Preprocessing Pipeline
Daily Transactions Dataset (prasad22)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os


class Dataset1Preprocessor:
    """
    Preprocessing pipeline for Dataset 1 (Daily Transactions Dataset).
    
    Columns:
    - Date: Transaction date
    - Mode: Payment mode (Cash, Bank Account, etc.)
    - Category: Transaction category (Transportation, Food, etc.)
    - Subcategory: More detailed classification
    - Note: Additional transaction-related notes
    - Amount: Transaction amount
    - Income/Expense: Specifies if transaction was income or expense
    - Currency: Transaction currency (e.g., INR)
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.categorical_columns = ['Mode', 'Category', 'Subcategory', 'Income/Expense', 'Currency']
        self.numerical_columns = ['Amount']
        self.columns_to_drop = ['Date', 'Note']  # Drop for ML, keep for reference
        
    def fit(self, df: pd.DataFrame):
        """
        Fit the preprocessing pipeline on the dataset.
        
        Args:
            df: Input DataFrame with raw data
        """
        df = df.copy()
        
        # Normalize column names
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Handle missing values in Subcategory and Note
        if 'subcategory' in df.columns:
            df['subcategory'].fillna('Not Specified', inplace=True)
        if 'note' in df.columns:
            df['note'].fillna('No Note', inplace=True)
        
        # Fit label encoders for categorical columns
        for col in self.categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Fit scaler on numerical columns
        numerical_data = df[self.numerical_columns].values
        self.scaler.fit(numerical_data)
        
        # Determine feature columns (categorical + numerical)
        self.feature_columns = self.categorical_columns + self.numerical_columns
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataset using fitted preprocessing pipeline.
        
        Args:
            df: Input DataFrame with raw data
            
        Returns:
            Transformed DataFrame ready for ML
        """
        df = df.copy()
        
        # Normalize column names
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Handle missing values
        if 'subcategory' in df.columns:
            df['subcategory'].fillna('Not Specified', inplace=True)
        if 'note' in df.columns:
            df['note'].fillna('No Note', inplace=True)
        
        # Encode categorical columns
        for col in self.categorical_columns:
            if col in df.columns and col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Scale numerical columns
        if len(self.numerical_columns) > 0:
            numerical_data = df[self.numerical_columns].values
            scaled_data = self.scaler.transform(numerical_data)
            for i, col in enumerate(self.numerical_columns):
                df[col] = scaled_data[:, i]
        
        # Select only feature columns
        result_df = df[self.feature_columns].copy()
        
        return result_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Input DataFrame with raw data
            
        Returns:
            Transformed DataFrame ready for ML
        """
        return self.fit(df).transform(df)
    
    def save_pipeline(self, save_dir: str):
        """
        Save the preprocessing pipeline components.
        
        Args:
            save_dir: Directory to save pipeline components
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save label encoders
        joblib.dump(self.label_encoders, os.path.join(save_dir, 'dataset1_label_encoders.pkl'))
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(save_dir, 'dataset1_scaler.pkl'))
        
        # Save feature columns
        joblib.dump(self.feature_columns, os.path.join(save_dir, 'dataset1_feature_columns.pkl'))
        
        # Save column mappings
        joblib.dump(self.categorical_columns, os.path.join(save_dir, 'dataset1_categorical_columns.pkl'))
        joblib.dump(self.numerical_columns, os.path.join(save_dir, 'dataset1_numerical_columns.pkl'))
    
    def load_pipeline(self, save_dir: str):
        """
        Load the preprocessing pipeline components.
        
        Args:
            save_dir: Directory containing saved pipeline components
        """
        self.label_encoders = joblib.load(os.path.join(save_dir, 'dataset1_label_encoders.pkl'))
        self.scaler = joblib.load(os.path.join(save_dir, 'dataset1_scaler.pkl'))
        self.feature_columns = joblib.load(os.path.join(save_dir, 'dataset1_feature_columns.pkl'))
        self.categorical_columns = joblib.load(os.path.join(save_dir, 'dataset1_categorical_columns.pkl'))
        self.numerical_columns = joblib.load(os.path.join(save_dir, 'dataset1_numerical_columns.pkl'))
        
        return self
    
    def get_feature_importance_names(self) -> list:
        """
        Get the names of features after preprocessing.
        
        Returns:
            List of feature names
        """
        return self.feature_columns if self.feature_columns else []
