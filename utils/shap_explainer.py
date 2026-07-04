"""
SHAP Model Explainability Module
Provides model explainability using SHAP values
"""

import shap
import numpy as np
import pandas as pd
import joblib
import os
from config import Config
import logging
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based model explainability for AML detection
    """
    
    def __init__(self, model, feature_names=None):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained ML model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """
        Initialize appropriate SHAP explainer based on model type
        """
        try:
            # Try TreeExplainer for tree-based models (Random Forest, XGBoost)
            if hasattr(self.model, 'estimators_') or 'xgboost' in str(type(self.model)).lower():
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("Initialized TreeExplainer for model")
            # Try KernelExplainer for other models (SVM, etc.)
            else:
                # For SVM and other models, use KernelExplainer
               # We'll need sample data for initialization
                self.explainer_type = 'kernel'
                logger.info("Will use KernelExplainer for model")
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {str(e)}")
            self.explainer = None
    
    def explain_single_prediction(self, features, feature_names=None):
        """
        Explain a single prediction using SHAP
        
        Args:
            features: Feature array (1D or 2D)
            feature_names: List of feature names (optional)
            
        Returns:
            dict: SHAP explanation results
        """
        if not self.explainer:
            return {'error': 'SHAP explainer not initialized'}
        
        # Ensure features is 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        try:
            # Use appropriate explainer
            if hasattr(self, 'explainer_type') and self.explainer_type == 'kernel':
                # For KernelExplainer, we need background data
                # Use the feature itself as background (not ideal but works)
                explainer = shap.KernelExplainer(self.model.predict, features)
                shap_values = explainer.shap_values(features)[0]
            else:
                shap_values = self.explainer.shap_values(features)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take first class for binary classification
            
            # Get feature names
            if feature_names is None:
                feature_names = self.feature_names
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(len(shap_values))]
            
            # Create feature importance dictionary
            feature_importance = dict(zip(feature_names, shap_values[0]))
            
            # Sort by absolute value
            sorted_importance = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            return {
                'shap_values': shap_values[0].tolist(),
                'feature_importance': dict(sorted_importance),
                'top_features': sorted_importance[:10],
                'base_value': float(self.explainer.expected_value[0]) if hasattr(self.explainer, 'expected_value') else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {str(e)}")
            return {'error': str(e)}
    
    def generate_force_plot(self, features, feature_names=None, save_path=None):
        """
        Generate SHAP force plot for a single prediction
        
        Args:
            features: Feature array
            feature_names: List of feature names
            save_path: Path to save the plot
            
        Returns:
            dict: Plot data (HTML or base64)
        """
        if not self.explainer:
            return {'error': 'SHAP explainer not initialized'}
        
        # Ensure features is 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        try:
            # Get SHAP values
            if hasattr(self, 'explainer_type') and self.explainer_type == 'kernel':
                explainer = shap.KernelExplainer(self.model.predict, features)
                shap_values = explainer.shap_values(features)[0]
            else:
                shap_values = self.explainer.shap_values(features)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Get feature names
            if feature_names is None:
                feature_names = self.feature_names
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(features.shape[1])]
            
            # Create force plot
            plt.figure()
            shap.force_plot(
                self.explainer.expected_value[0] if hasattr(self.explainer, 'expected_value') else 0,
                shap_values[0],
                features[0],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            
            # Save or convert to base64
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close()
                return {'path': save_path}
            else:
                # Convert to base64
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                plt.close()
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                return {'image': img_base64}
                
        except Exception as e:
            logger.error(f"Error generating force plot: {str(e)}")
            return {'error': str(e)}
    
    def generate_waterfall_plot(self, features, feature_names=None, save_path=None):
        """
        Generate SHAP waterfall plot for a single prediction
        
        Args:
            features: Feature array
            feature_names: List of feature names
            save_path: Path to save the plot
            
        Returns:
            dict: Plot data (HTML or base64)
        """
        if not self.explainer:
            return {'error': 'SHAP explainer not initialized'}
        
        # Ensure features is 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        try:
            # Get SHAP values
            if hasattr(self, 'explainer_type') and self.explainer_type == 'kernel':
                explainer = shap.KernelExplainer(self.model.predict, features)
                shap_values = explainer.shap_values(features)[0]
            else:
                shap_values = self.explainer.shap_values(features)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Get feature names
            if feature_names is None:
                feature_names = self.feature_names
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(features.shape[1])]
            
            # Create waterfall plot
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=self.explainer.expected_value[0] if hasattr(self.explainer, 'expected_value') else 0,
                    data=features[0],
                    feature_names=feature_names
                ),
                show=False
            )
            
            # Save or convert to base64
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close()
                return {'path': save_path}
            else:
                # Convert to base64
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                plt.close()
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                return {'image': img_base64}
                
        except Exception as e:
            logger.error(f"Error generating waterfall plot: {str(e)}")
            return {'error': str(e)}
    
    def generate_summary_plot(self, X, feature_names=None, save_path=None):
        """
        Generate SHAP summary plot for multiple predictions
        
        Args:
            X: Feature matrix (2D array)
            feature_names: List of feature names
            save_path: Path to save the plot
            
        Returns:
            dict: Plot data (HTML or base64)
        """
        if not self.explainer:
            return {'error': 'SHAP explainer not initialized'}
        
        try:
            # Get SHAP values
            if hasattr(self, 'explainer_type') and self.explainer_type == 'kernel':
                explainer = shap.KernelExplainer(self.model.predict, X[:100])  # Use subset for kernel
                shap_values = explainer.shap_values(X[:100])[0]
            else:
                shap_values = self.explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Get feature names
            if feature_names is None:
                feature_names = self.feature_names
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
            
            # Create summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values,
                X[:100] if hasattr(self, 'explainer_type') else X,
                feature_names=feature_names,
                show=False
            )
            
            # Save or convert to base64
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close()
                return {'path': save_path}
            else:
                # Convert to base64
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                plt.close()
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                return {'image': img_base64}
                
        except Exception as e:
            logger.error(f"Error generating summary plot: {str(e)}")
            return {'error': str(e)}


def get_shap_explainer(model, feature_names=None):
    """
    Factory function to get SHAP explainer for a model
    
    Args:
        model: Trained ML model
        feature_names: List of feature names
        
    Returns:
        SHAPExplainer instance
    """
    return SHAPExplainer(model, feature_names)
