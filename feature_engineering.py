"""
Feature engineering module for Money Laundering Detection Platform
Extracts and creates features for ML models and risk analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature extraction and engineering for AML detection"""
    
    def __init__(self):
        """Initialize feature engineer"""
        self.risk_scores = {}
        self.merchant_risk_map = {}
        self.bank_risk_map = {}
        self.device_risk_map = {}
        self.location_risk_map = {}
    
    def extract_temporal_features(self, df):
        """
        Extract temporal features from timestamp
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Data with temporal features
        """
        df = df.copy()
        
        # Try to parse timestamp if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Extract time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Time risk: transactions during odd hours (night time)
            df['time_risk'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
            
            logger.info("Temporal features extracted")
        
        return df
    
    def calculate_transaction_velocity(self, df, window_minutes=60):
        """
        Calculate transaction velocity for each sender
        
        Args:
            df: Input DataFrame
            window_minutes: Time window in minutes
            
        Returns:
            DataFrame: Data with velocity features
        """
        df = df.copy()
        
        if 'timestamp' in df.columns and 'sender' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.sort_values('timestamp')
            
            # Calculate number of transactions in time window
            velocity = []
            for idx, row in df.iterrows():
                sender = row['sender']
                current_time = row['timestamp']
                
                if pd.isna(current_time):
                    velocity.append(0)
                    continue
                
                # Count transactions in window
                window_start = current_time - timedelta(minutes=window_minutes)
                window_transactions = df[
                    (df['sender'] == sender) & 
                    (df['timestamp'] >= window_start) & 
                    (df['timestamp'] <= current_time)
                ]
                
                velocity.append(len(window_transactions))
            
            df['transaction_velocity'] = velocity
            
            # Normalize velocity (0-100)
            if df['transaction_velocity'].max() > 0:
                df['velocity_risk'] = (df['transaction_velocity'] / df['transaction_velocity'].max() * 100).clip(0, 100)
            else:
                df['velocity_risk'] = 0
            
            logger.info("Transaction velocity calculated")
        
        return df
    
    def calculate_amount_features(self, df):
        """
        Calculate amount-based features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Data with amount features
        """
        df = df.copy()
        
        if 'amount' in df.columns:
            # Log transform amount
            df['amount_log'] = np.log1p(df['amount'])
            
            # Amount risk based on percentiles
            if len(df) > 0:
                amount_percentiles = df['amount'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
                
                def get_amount_risk(amount):
                    if amount >= amount_percentiles[0.99]:
                        return 100
                    elif amount >= amount_percentiles[0.95]:
                        return 80
                    elif amount >= amount_percentiles[0.9]:
                        return 60
                    elif amount >= amount_percentiles[0.75]:
                        return 40
                    elif amount >= amount_percentiles[0.5]:
                        return 20
                    else:
                        return 10
                
                df['amount_risk'] = df['amount'].apply(get_amount_risk)
            
            logger.info("Amount features calculated")
        
        return df
    
    def calculate_frequency_features(self, df):
        """
        Calculate frequency-based features for senders and receivers
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Data with frequency features
        """
        df = df.copy()
        
        if 'sender' in df.columns:
            # Sender frequency
            sender_counts = df['sender'].value_counts()
            df['sender_frequency'] = df['sender'].map(sender_counts)
            
            # Normalize sender frequency risk (higher frequency = higher risk)
            if df['sender_frequency'].max() > 0:
                df['sender_frequency_risk'] = (df['sender_frequency'] / df['sender_frequency'].max() * 100).clip(0, 100)
            else:
                df['sender_frequency_risk'] = 0
        
        if 'receiver' in df.columns:
            # Receiver frequency
            receiver_counts = df['receiver'].value_counts()
            df['receiver_frequency'] = df['receiver'].map(receiver_counts)
            
            # Normalize receiver frequency risk
            if df['receiver_frequency'].max() > 0:
                df['receiver_frequency_risk'] = (df['receiver_frequency'] / df['receiver_frequency'].max() * 100).clip(0, 100)
            else:
                df['receiver_frequency_risk'] = 0
        
        logger.info("Frequency features calculated")
        return df
    
    def calculate_entity_risk_scores(self, df):
        """
        Calculate risk scores for entities (banks, merchants, devices, locations)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Data with entity risk scores
        """
        df = df.copy()
        
        # Bank risk based on fraud rate
        if 'bank' in df.columns and 'prediction' in df.columns:
            bank_fraud_rate = df.groupby('bank')['prediction'].mean()
            self.bank_risk_map = (bank_fraud_rate * 100).to_dict()
            df['bank_risk'] = df['bank'].map(self.bank_risk_map).fillna(10)
        
        # Merchant risk
        if 'merchant' in df.columns and 'prediction' in df.columns:
            merchant_fraud_rate = df.groupby('merchant')['prediction'].mean()
            self.merchant_risk_map = (merchant_fraud_rate * 100).to_dict()
            df['merchant_risk'] = df['merchant'].map(self.merchant_risk_map).fillna(10)
        
        # Device risk
        if 'device' in df.columns and 'prediction' in df.columns:
            device_fraud_rate = df.groupby('device')['prediction'].mean()
            self.device_risk_map = (device_fraud_rate * 100).to_dict()
            df['device_risk'] = df['device'].map(self.device_risk_map).fillna(10)
        
        # Location risk
        if 'location' in df.columns and 'prediction' in df.columns:
            location_fraud_rate = df.groupby('location')['prediction'].mean()
            self.location_risk_map = (location_fraud_rate * 100).to_dict()
            df['location_risk'] = df['location'].map(self.location_risk_map).fillna(10)
        
        logger.info("Entity risk scores calculated")
        return df
    
    def calculate_network_features(self, df):
        """
        Calculate network-based features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Data with network features
        """
        df = df.copy()
        
        if 'sender' in df.columns and 'receiver' in df.columns:
            # Number of unique receivers per sender
            sender_receiver_count = df.groupby('sender')['receiver'].nunique()
            df['sender_unique_receivers'] = df['sender'].map(sender_receiver_count)
            
            # Number of unique senders per receiver
            receiver_sender_count = df.groupby('receiver')['sender'].nunique()
            df['receiver_unique_senders'] = df['receiver'].map(receiver_sender_count)
            
            # Network risk (many connections = higher risk)
            max_connections = max(df['sender_unique_receivers'].max(), df['receiver_unique_senders'].max())
            if max_connections > 0:
                df['network_risk'] = (
                    (df['sender_unique_receivers'] + df['receiver_unique_senders']) / max_connections * 100
                ).clip(0, 100)
            else:
                df['network_risk'] = 0
            
            logger.info("Network features calculated")
        
        return df
    
    def calculate_historical_pattern_risk(self, df):
        """
        Calculate historical pattern risk based on transaction patterns
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Data with historical pattern risk
        """
        df = df.copy()
        
        if 'sender' in df.columns and 'amount' in df.columns:
            # Calculate average amount per sender
            sender_avg_amount = df.groupby('sender')['amount'].mean()
            df['sender_avg_amount'] = df['sender'].map(sender_avg_amount)
            
            # Deviation from sender's average
            df['amount_deviation'] = abs(df['amount'] - df['sender_avg_amount']) / (df['sender_avg_amount'] + 1)
            
            # Historical pattern risk based on deviation
            df['historical_pattern_risk'] = (df['amount_deviation'] * 100).clip(0, 100)
            
            logger.info("Historical pattern risk calculated")
        
        return df
    
    def calculate_spider_chart_features(self, row):
        """
        Calculate features for spider/radar chart visualization
        
        Args:
            row: Single transaction row
            
        Returns:
           .dict: Spider chart feature values (0-100)
        """
        spider_features = {
            'Transaction Amount': row.get('amount_risk', 50),
            'Frequency': row.get('sender_frequency_risk', 50),
            'Sender Risk': row.get('sender_frequency_risk', 50),
            'Receiver Risk': row.get('receiver_frequency_risk', 50),
            'Merchant Risk': row.get('merchant_risk', 20),
            'Bank Risk': row.get('bank_risk', 20),
            'Device Risk': row.get('device_risk', 20),
            'Location Risk': row.get('location_risk', 20),
            'Weekend Activity': row.get('is_weekend', 0) * 100,
            'Time Risk': row.get('time_risk', 0) * 100,
            'Network Risk': row.get('network_risk', 30),
            'Transaction Velocity': row.get('velocity_risk', 30),
            'Historical Pattern': row.get('historical_pattern_risk', 30),
            'AML Score': row.get('probability', 0) * 100
        }
        
        # Ensure all values are between 0 and 100
        spider_features = {k: min(max(v, 0), 100) for k, v in spider_features.items()}
        
        return spider_features
    
    def calculate_overall_risk_score(self, df):
        """
        Calculate overall risk score combining all risk factors
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Data with overall risk score
        """
        df = df.copy()
        
        risk_columns = [
            'amount_risk', 'sender_frequency_risk', 'receiver_frequency_risk',
            'bank_risk', 'merchant_risk', 'device_risk', 'location_risk',
            'time_risk', 'network_risk', 'velocity_risk', 'historical_pattern_risk'
        ]
        
        available_risk_columns = [col for col in risk_columns if col in df.columns]
        
        if available_risk_columns:
            # Calculate weighted average
            weights = {
                'amount_risk': 0.2,
                'sender_frequency_risk': 0.1,
                'receiver_frequency_risk': 0.1,
                'bank_risk': 0.1,
                'merchant_risk': 0.1,
                'device_risk': 0.1,
                'location_risk': 0.1,
                'time_risk': 0.05,
                'network_risk': 0.1,
                'velocity_risk': 0.05,
                'historical_pattern_risk': 0.1
            }
            
            total_weight = sum(weights.get(col, 0) for col in available_risk_columns)
            weighted_sum = sum(df[col] * weights.get(col, 0) for col in available_risk_columns)
            
            df['calculated_risk_score'] = (weighted_sum / total_weight).clip(0, 100)
        else:
            df['calculated_risk_score'] = 50  # Default medium risk
        
        logger.info("Overall risk score calculated")
        return df
    
    def engineer_features(self, df, for_prediction=True):
        """
        Complete feature engineering pipeline
        
        Args:
            df: Input DataFrame
            for_prediction: Whether this is for prediction (vs training)
            
        Returns:
            DataFrame: Data with engineered features
        """
        logger.info("Starting feature engineering")
        
        # Extract temporal features
        df = self.extract_temporal_features(df)
        
        # Calculate transaction velocity
        df = self.calculate_transaction_velocity(df)
        
        # Calculate amount features
        df = self.calculate_amount_features(df)
        
        # Calculate frequency features
        df = self.calculate_frequency_features(df)
        
        # Calculate entity risk scores (only if prediction column exists)
        if 'prediction' in df.columns or not for_prediction:
            df = self.calculate_entity_risk_scores(df)
        
        # Calculate network features
        df = self.calculate_network_features(df)
        
        # Calculate historical pattern risk
        df = self.calculate_historical_pattern_risk(df)
        
        # Calculate overall risk score
        df = self.calculate_overall_risk_score(df)
        
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df
    
    def get_feature_importance(self, model, feature_names):
        """
        Get feature importance from trained model
        
        Args:
            model: Trained ML model
            feature_names: List of feature names
            
        Returns:
            dict: Feature importance scores
        """
        importance = {}
        
        if hasattr(model, 'feature_importances_'):
            # Random Forest, XGBoost
            for name, score in zip(feature_names, model.feature_importances_):
                importance[name] = score
        elif hasattr(model, 'coef_'):
            # SVM, Logistic Regression
            for name, score in zip(feature_names, np.abs(model.coef_[0])):
                importance[name] = score
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance
