"""
Admin Blueprint
Handles admin panel with system logs and model performance
"""

from flask import Blueprint, render_template, session, jsonify
from database import db
from prediction import PredictionEngine
import logging

logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


@admin_bp.route('/admin')
def admin():
    """Admin panel page"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    # Check if user is admin
    if session.get('role') != 'admin':
        return render_template('error.html', error='Access denied. Admin only.')
    
    try:
        # Get model performance data
        model_performance = db.get_model_performance()
        
        # Get system logs
        system_logs = []
        # This would need to be implemented in database.py
        # For now, return empty list
        
        # Get dataset statistics
        dataset_stats = {
            'dataset1': {'total_uploads': 0, 'total_transactions': 0},
            'dataset2': {'total_uploads': 0, 'total_transactions': 0}
        }
        
        return render_template('admin.html', 
                              model_performance=model_performance,
                              system_logs=system_logs,
                              dataset_stats=dataset_stats)
    except Exception as e:
        logger.error(f"Admin error: {str(e)}")
        return render_template('error.html', error=str(e))


@admin_bp.route('/api/system-logs')
def system_logs_api():
    """API endpoint for system logs"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if session.get('role') != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        # Get system logs from database
        # This would need to be implemented in database.py
        logs = []
        return jsonify(logs)
    except Exception as e:
        logger.error(f"System logs API error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/api/model-performance')
def model_performance_api():
    """API endpoint for model performance"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if session.get('role') != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        # Get model performance for both datasets
        performance_data = {}
        for dataset_type in ['dataset1', 'dataset2']:
            try:
                pred_engine = PredictionEngine(dataset_type)
                status = pred_engine.get_model_status()
                performance_data[dataset_type] = status
            except Exception as e:
                logger.error(f"Error loading model performance for {dataset_type}: {str(e)}")
                performance_data[dataset_type] = {}
        
        return jsonify(performance_data)
    except Exception as e:
        logger.error(f"Model performance API error: {str(e)}")
        return jsonify({'error': str(e)}), 500
