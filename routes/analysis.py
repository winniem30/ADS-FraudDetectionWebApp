"""
Analysis Blueprint
Handles transaction analysis, search, and model comparison
"""

from flask import Blueprint, render_template, request, session, jsonify
from database import db
from prediction import PredictionEngine
import logging

logger = logging.getLogger(__name__)

analysis_bp = Blueprint('analysis', __name__, url_prefix='/analysis')


@analysis_bp.route('/transactions')
def transactions():
    """Transaction list page with filters"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    page = request.args.get('page', 1, type=int)
    filters = {
        'risk_level': request.args.get('risk_level'),
        'prediction': request.args.get('prediction'),
        'min_amount': request.args.get('min_amount', type=float) if request.args.get('min_amount') else None,
        'max_amount': request.args.get('max_amount', type=float) if request.args.get('max_amount') else None
    }
    
    try:
        transactions = db.get_transactions(limit=20, offset=(page-1)*20, filters=filters)
        return render_template('transactions.html', transactions=transactions, page=page, filters=filters)
    except Exception as e:
        logger.error(f"Transactions error: {str(e)}")
        return render_template('error.html', error=str(e))


@analysis_bp.route('/transaction/<int:transaction_id>')
def transaction_detail(transaction_id):
    """Transaction detail page with spider chart and SHAP explanation"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    try:
        transaction = db.get_transaction(transaction_id)
        if not transaction:
            return render_template('error.html', error='Transaction not found')
        
        # Get spider chart data
        from visualization import VisualizationEngine
        viz_engine = VisualizationEngine()
        
        # Generate spider chart features based on dataset type
        dataset_type = transaction.get('dataset_type', 'dataset1')
        spider_features = generate_spider_features(transaction, dataset_type)
        
        spider_chart = viz_engine.create_spider_chart(
            spider_features, 
            dataset_type=dataset_type,
            title=f"Transaction {transaction_id} Risk Analysis"
        )
        
        # Get SHAP explanation if available
        shap_explanation = None
        try:
            from utils.shap_explainer import get_shap_explainer
            pred_engine = PredictionEngine(dataset_type)
            if 'random_forest' in pred_engine.models:
                explainer = get_shap_explainer(pred_engine.models['random_forest'])
                # Extract features from transaction
                # This would need the actual feature values
                # For now, skip SHAP as it requires feature extraction
                pass
        except Exception as e:
            logger.warning(f"SHAP explanation not available: {str(e)}")
        
        return render_template('transaction_detail.html', 
                              transaction=transaction, 
                              spider_chart=spider_chart,
                              shap_explanation=shap_explanation)
    except Exception as e:
        logger.error(f"Transaction detail error: {str(e)}")
        return render_template('error.html', error=str(e))


@analysis_bp.route('/search')
def search():
    """Search page"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    query = request.args.get('q', '')
    field = request.args.get('field', 'all')
    
    if query:
        try:
            results = db.search_transactions(query, field)
            return render_template('search.html', results=results, query=query, field=field)
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return render_template('error.html', error=str(e))
    
    return render_template('search.html', results=[], query='', field='all')


@analysis_bp.route('/network-analysis')
def network_analysis():
    """Network analysis page"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    try:
        # Get network analysis data
        # This would generate network graph from transactions
        network_data = {
            'node_count': 0,
            'edge_count': 0,
            'html': None
        }
        
        return render_template('network_analysis.html', network_data=network_data)
    except Exception as e:
        logger.error(f"Network analysis error: {str(e)}")
        return render_template('error.html', error=str(e))


@analysis_bp.route('/model-comparison')
def model_comparison():
    """Model comparison page"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    try:
        # Get model status for both datasets
        model_status = {}
        for dataset_type in ['dataset1', 'dataset2']:
            try:
                pred_engine = PredictionEngine(dataset_type)
                status = pred_engine.get_model_status()
                for model_name, status_info in status.items():
                    model_status[f"{dataset_type}_{model_name}"] = status_info
            except Exception as e:
                logger.error(f"Error loading models for {dataset_type}: {str(e)}")
        
        return render_template('model_comparison.html', model_status=model_status)
    except Exception as e:
        logger.error(f"Model comparison error: {str(e)}")
        return render_template('error.html', error=str(e))


@analysis_bp.route('/api/model-comparison', methods=['POST'])
def model_comparison_api():
    """API endpoint for model comparison"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = request.get_json()
        upload_id = data.get('upload_id')
        dataset_type = data.get('dataset_type', 'dataset1')
        
        # Get transactions for upload
        # This would need to be implemented in database.py
        # For now, return empty comparison
        comparison_results = {}
        
        return jsonify(comparison_results)
    except Exception as e:
        logger.error(f"Model comparison API error: {str(e)}")
        return jsonify({'error': str(e)}), 500


def generate_spider_features(transaction, dataset_type):
    """Generate spider chart features from transaction data"""
    features = {}
    
    if dataset_type == 'dataset1':
        # Dataset 1 spider features
        features['Amount'] = min(100, max(0, abs(transaction.get('amount', 0) / 10000 * 100)))
        features['Mode'] = 50  # Default
        features['Category'] = 50  # Default
        features['Subcategory'] = 50  # Default
        features['Income/Expense'] = 50  # Default
        features['Frequency'] = min(100, max(0, transaction.get('probability', 0) * 100))
        features['Time'] = 50  # Default
        features['Risk Score'] = transaction.get('risk_score', 0)
        features['Behavior Score'] = min(100, max(0, transaction.get('confidence_score', 0)))
        features['Transaction Pattern'] = min(100, max(0, transaction.get('probability', 0) * 100))
    else:
        # Dataset 2 spider features
        features['Transaction Amount'] = min(100, max(0, abs(transaction.get('amount', 0) / 10000 * 100)))
        features['Customer Risk'] = 50  # Default
        features['Transaction Type'] = 50  # Default
        features['Device Risk'] = 50  # Default
        features['Merchant Risk'] = 50  # Default
        features['Location Risk'] = 50  # Default
        features['Frequency'] = min(100, max(0, transaction.get('probability', 0) * 100))
        features['Time Risk'] = 50  # Default
        features['Network Risk'] = 50  # Default
        features['Historical Behaviour'] = 50  # Default
        features['Velocity'] = 50  # Default
        features['AML Score'] = transaction.get('risk_score', 0)
    
    return features
