"""
Data preprocessing module for Money Laundering Detection Platform
Handles data cleaning, transformation, and preparation for ML models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self):
        """Initialize preprocessor with encoders and scalers"""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.categorical_columns = []
        self.numerical_columns = []
    
    def load_data(self, file_path):
        """
        Load data from CSV or Excel file
        
        Args:
            file_path: Path to the data file
            
        Returns:
            DataFrame: Loaded data
        """
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
            
            logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self, df):
        """
        Validate the dataset structure and content
        
        Args:
            df: Input DataFrame
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if df is None or df.empty:
            return False, "Dataset is empty"
        
        # Check for minimum required columns
        required_columns = ['amount', 'sender', 'receiver']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Check for sufficient data
        if len(df) < 10:
            return False, "Dataset must contain at least 10 rows"
        
        return True, "Data validation passed"
    
    def clean_data(self, df):
        """
        Clean the dataset by handling missing values and duplicates
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Cleaned data
        """
        df_cleaned = df.copy()
        
        # Remove duplicates
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        duplicates_removed = initial_rows - len(df_cleaned)
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        # Handle missing values
        for column in df_cleaned.columns:
            if df_cleaned[column].isnull().sum() > 0:
                if df_cleaned[column].dtype in ['int64', 'float64']:
                    # Fill numerical columns with median
                    df_cleaned[column].fillna(df_cleaned[column].median(), inplace=True)
                else:
                    # Fill categorical columns with mode
                    df_cleaned[column].fillna(df_cleaned[column].mode()[0] if not df_cleaned[column].mode().empty else 'Unknown', inplace=True)
        
        logger.info(f"Data cleaned: {len(df_cleaned)} rows remaining")
        return df_cleaned
    
    def remove_columns(self, df, columns_to_remove):
        """
        Remove specified columns from DataFrame
        
        Args:
            df: Input DataFrame
            columns_to_remove: List of column names to remove
            
        Returns:
            DataFrame: Data with columns removed
        """
        # Remove columns that exist
        existing_columns = [col for col in columns_to_remove if col in df.columns]
        
        if existing_columns:
            df = df.drop(columns=existing_columns)
            logger.info(f"Removed columns: {', '.join(existing_columns)}")
        
        return df
    
    def identify_column_types(self, df):
        """
        Identify categorical and numerical columns
        
        Args:
            df: Input DataFrame
        """
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        logger.info(f"Categorical columns: {len(self.categorical_columns)}")
        logger.info(f"Numerical columns: {len(self.numerical_columns)}")
    
    def encode_categorical_features(self, df, fit=True):
        """
        Encode categorical features using label encoding
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders (True for training, False for prediction)
            
        Returns:
            DataFrame: Data with encoded categorical features
        """
        df_encoded = df.copy()
        
        for column in self.categorical_columns:
            if column in df_encoded.columns:
                if fit:
                    # Create new encoder and fit
                    le = LabelEncoder()
                    df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
                    self.label_encoders[column] = le
                else:
                    # Use existing encoder, handle unseen categories
                    if column in self.label_encoders:
                        le = self.label_encoders[column]
                        # Handle unseen categories by assigning a new label
                        unique_values = set(df_encoded[column].astype(str).unique())
                        known_classes = set(le.classes_)
                        unseen_values = unique_values - known_classes
                        
                        if unseen_values:
                            # Add unseen values to encoder
                            new_classes = list(le.classes_) + list(unseen_values)
                            le.classes_ = np.array(new_classes)
                        
                        df_encoded[column] = le.transform(df_encoded[column].astype(str))
                    else:
                        # If encoder doesn't exist, create one
                        le = LabelEncoder()
                        df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
                        self.label_encoders[column] = le
        
        return df_encoded
    
    def scale_numerical_features(self, df, fit=True):
        """
        Scale numerical features using StandardScaler
        
        Args:
            df: Input DataFrame
            fit: Whether to fit scaler (True for training, False for prediction)
            
        Returns:
            DataFrame: Data with scaled numerical features
        """
        df_scaled = df.copy()
        
        numerical_cols_to_scale = [col for col in self.numerical_columns if col in df_scaled.columns]
        
        if numerical_cols_to_scale:
            if fit:
                df_scaled[numerical_cols_to_scale] = self.scaler.fit_transform(df_scaled[numerical_cols_to_scale])
            else:
                df_scaled[numerical_cols_to_scale] = self.scaler.transform(df_scaled[numerical_cols_to_scale])
            
            logger.info(f"Scaled {len(numerical_cols_to_scale)} numerical features")
        
        return df_scaled
    
    def align_columns(self, df, expected_columns=None):
        """
        Align DataFrame columns with expected columns from training
        
        Args:
            df: Input DataFrame
            expected_columns: List of expected column names
            
        Returns:
            DataFrame: Data with aligned columns
        """
        if expected_columns is None:
            # If no expected columns, use current columns as reference
            self.feature_columns = df.columns.tolist()
            return df
        
        # Add missing columns with zeros
        missing_columns = set(expected_columns) - set(df.columns)
        for col in missing_columns:
            df[col] = 0
            logger.info(f"Added missing column: {col}")
        
        # Remove extra columns
        extra_columns = set(df.columns) - set(expected_columns)
        if extra_columns:
            df = df.drop(columns=list(extra_columns))
            logger.info(f"Removed extra columns: {', '.join(extra_columns)}")
        
        # Reorder columns to match expected order
        df = df[expected_columns]
        
        return df
    
    def preprocess_for_prediction(self, df, model_type='random_forest'):
        """
        Complete preprocessing pipeline for prediction
        
        Args:
            df: Input DataFrame
            model_type: Type of model being used
            
        Returns:
            DataFrame: Preprocessed data ready for prediction
        """
        logger.info(f"Starting preprocessing for {model_type} model")
        
        # Step 1: Remove transaction ID and timestamp if present
        columns_to_remove = ['transaction_id', 'timestamp', 'date', 'time']
        df = self.remove_columns(df, columns_to_remove)
        
        # Step 2: Clean data
        df = self.clean_data(df)
        
        # Step 3: Identify column types
        self.identify_column_types(df)
        
        # Step 4: Encode categorical features
        df = self.encode_categorical_features(df, fit=False)
        
        # Step 5: Scale numerical features
        df = self.scale_numerical_features(df, fit=False)
        
        # Step 6: Align columns with training data
        if self.feature_columns:
            df = self.align_columns(df, self.feature_columns)
        else:
            self.feature_columns = df.columns.tolist()
        
        logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        return df
    
    def preprocess_for_training(self, df):
        """
        Complete preprocessing pipeline for training
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Preprocessed data ready for training
        """
        logger.info("Starting preprocessing for training")
        
        # Step 1: Remove transaction ID and timestamp if present
        columns_to_remove = ['transaction_id', 'timestamp', 'date', 'time']
        df = self.remove_columns(df, columns_to_remove)
        
        # Step 2: Clean data
        df = self.clean_data(df)
        
        # Step 3: Identify column types
        self.identify_column_types(df)
        
        # Step 4: Encode categorical features
        df = self.encode_categorical_features(df, fit=True)
        
        # Step 5: Scale numerical features
        df = self.scale_numerical_features(df, fit=True)
        
        # Step 6: Store feature columns
        self.feature_columns = df.columns.tolist()
        
        logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        return df
    
    def save_preprocessor(self, file_path):
        """
        Save preprocessor state to file
        
        Args:
            file_path: Path to save the preprocessor
        """
        preprocessor_state = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(preprocessor_state, f)
        
        logger.info(f"Preprocessor saved to {file_path}")
    
    def load_preprocessor(self, file_path):
        """
        Load preprocessor state from file
        
        Args:
            file_path: Path to load the preprocessor from
        """
        with open(file_path, 'rb') as f:
            preprocessor_state = pickle.load(f)
        
        self.label_encoders = preprocessor_state['label_encoders']
        self.scaler = preprocessor_state['scaler']
        self.feature_columns = preprocessor_state['feature_columns']
        self.categorical_columns = preprocessor_state['categorical_columns']
        self.numerical_columns = preprocessor_state['numerical_columns']
        
        logger.info(f"Preprocessor loaded from {file_path}")
    
    def get_data_summary(self, df):
        """
        Generate summary statistics of the data
        
        Args:
            df: Input DataFrame
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'numerical_columns': len(self.numerical_columns) if self.numerical_columns else 0,
            'categorical_columns': len(self.categorical_columns) if self.categorical_columns else 0,
            'column_names': df.columns.tolist()
        }
        
        return summary
