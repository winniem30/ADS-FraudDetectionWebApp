"""
Routes module for Money Laundering Detection Platform
Contains all Flask routes and blueprints
"""

from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import os
import pandas as pd
from datetime import datetime
import json
from functools import wraps

from config import Config
from database import db
from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from prediction import PredictionEngine, ModelComparator
from visualization import VisualizationEngine

# Initialize engines
preprocessor = DataPreprocessor()
feature_engineer = FeatureEngineer()
prediction_engine = PredictionEngine()
model_comparator = ModelComparator(prediction_engine)
viz_engine = VisualizationEngine()

# Create blueprints
main_bp = Blueprint('main', __name__)
auth_bp = Blueprint('auth', __name__)
dashboard_bp = Blueprint('dashboard', __name__)
upload_bp = Blueprint('upload', __name__)
analysis_bp = Blueprint('analysis', __name__)
report_bp = Blueprint('report', __name__)
admin_bp = Blueprint('admin', __name__)


# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


# ==================== Main Routes ====================

@main_bp.route('/')
def index():
    """Home page - redirect to login or dashboard"""
    if 'user_id' in session:
        return redirect(url_for('dashboard.dashboard'))
    return redirect(url_for('auth.login'))


@main_bp.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models': prediction_engine.get_model_status()
    })


# ==================== Auth Routes ====================

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Simple authentication (in production, use proper password hashing)
        user = db.get_user(username)
        
        if user and user['password_hash'] == password:  # Simplified for demo
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            db.update_last_login(user['id'])
            return redirect(url_for('dashboard.dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')


@auth_bp.route('/logout')
def logout():
    """Logout"""
    session.clear()
    return redirect(url_for('auth.login'))


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        
        # Check if user exists
        if db.get_user(username):
            return render_template('register.html', error='Username already exists')
        
        # Create user
        db.insert_user(username, password, email)
        return redirect(url_for('auth.login'))
    
    return render_template('register.html')


# ==================== Dashboard Routes ====================

@dashboard_bp.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard"""
    stats = db.get_dashboard_stats()
    model_status = prediction_engine.get_model_status()
    alerts = db.get_alerts(unread_only=True, limit=5)
    
    return render_template('dashboard.html',
                          stats=stats,
                          model_status=model_status,
                          alerts=alerts,
                          user=session.get('username'))


@dashboard_bp.route('/api/dashboard-stats')
@login_required
def api_dashboard_stats():
    """API endpoint for dashboard statistics"""
    stats = db.get_dashboard_stats()
    return jsonify(stats)


@dashboard_bp.route('/api/alerts')
@login_required
def api_alerts():
    """API endpoint for alerts"""
    alerts = db.get_alerts(unread_only=False, limit=20)
    return jsonify([dict(alert) for alert in alerts])


@dashboard_bp.route('/api/alert/<int:alert_id>/read', methods=['POST'])
@login_required
def mark_alert_read(alert_id):
    """Mark alert as read"""
    db.mark_alert_read(alert_id)
    return jsonify({'status': 'success'})


# ==================== Upload Routes ====================

@upload_bp.route('/upload')
@login_required
def upload():
    """Upload page"""
    recent_uploads = db.get_dashboard_stats().get('recent_uploads', [])
    return render_template('upload.html', recent_uploads=recent_uploads)


@upload_bp.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    model_choice = request.form.get('model', 'random_forest')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(Config.UPLOAD_FOLDER, saved_filename)
        file.save(filepath)
        
        # Create upload record
        file_size = os.path.getsize(filepath)
        upload_id = db.insert_upload(saved_filename, filename, file_size)
        
        try:
            # Load and validate data
            df = preprocessor.load_data(filepath)
            is_valid, error_msg = preprocessor.validate_data(df)
            
            if not is_valid:
                db.update_upload(upload_id, len(df), 'failed', model_choice)
                return jsonify({'error': error_msg}), 400
            
            # Store original data for reference
            original_df = df.copy()
            
            # Preprocess data
            df_processed = preprocessor.preprocess_for_prediction(df, model_choice)
            
            # Feature engineering
            df_engineered = feature_engineer.engineer_features(df_processed)
            
            # Make predictions
            results = prediction_engine.generate_prediction_results(
                df_engineered, original_df, model_choice
            )
            
            # Store results in database
            for result in results:
                result['upload_id'] = upload_id
                db.insert_transaction(result)
            
            # Generate alerts for high-risk transactions
            high_risk_transactions = [r for r in results if r['risk_level'] in ['high', 'critical']]
            for tx in high_risk_transactions:
                db.insert_alert(
                    tx.get('id'),
                    'high_risk',
                    tx['risk_level'],
                    f"High risk transaction detected: {tx.get('transaction_id', 'N/A')} with risk score {tx['risk_score']:.2f}"
                )
            
            # Update upload record
            db.update_upload(upload_id, len(df), 'completed', model_choice)
            
            return jsonify({
                'status': 'success',
                'upload_id': upload_id,
                'total_transactions': len(results),
                'high_risk_count': len(high_risk_transactions),
                'results': results[:10]  # Return first 10 for preview
            })
            
        except Exception as e:
            db.update_upload(upload_id, 0, 'failed', model_choice)
            db.log_system_event('ERROR', 'upload', str(e), {'file': filename})
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400


@upload_bp.route('/upload/preview', methods=['POST'])
@login_required
def upload_preview():
    """Preview uploaded data before processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        df = preprocessor.load_data(file)
        summary = preprocessor.get_data_summary(df)
        
        return jsonify({
            'status': 'success',
            'summary': summary,
            'columns': df.columns.tolist(),
            'preview': df.head(5).to_dict('records')
        })
    
    return jsonify({'error': 'Invalid file type'}), 400


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


# ==================== Analysis Routes ====================

@analysis_bp.route('/transactions')
@login_required
def transactions():
    """Transactions list page"""
    page = request.args.get('page', 1, type=int)
    filters = {
        'risk_level': request.args.get('risk_level'),
        'prediction': request.args.get('prediction'),
        'model_used': request.args.get('model'),
        'min_amount': request.args.get('min_amount', type=float),
        'max_amount': request.args.get('max_amount', type=float)
    }
    
    # Remove None filters
    filters = {k: v for k, v in filters.items() if v is not None}
    
    transactions = db.get_transactions(
        limit=Config.ITEMS_PER_PAGE,
        offset=(page - 1) * Config.ITEMS_PER_PAGE,
        filters=filters
    )
    
    return render_template('transactions.html', 
                          transactions=transactions,
                          page=page,
                          filters=filters)


@analysis_bp.route('/transaction/<int:transaction_id>')
@login_required
def transaction_detail(transaction_id):
    """Transaction detail page with spider chart"""
    transaction = db.get_transaction(transaction_id)
    
    if not transaction:
        return render_template('error.html', error='Transaction not found'), 404
    
    # Convert to dict
    tx_dict = dict(transaction)
    
    # Generate spider chart features
    spider_features = {
        'Transaction Amount': tx_dict.get('risk_score', 50),
        'Frequency': 50,
        'Sender Risk': 50,
        'Receiver Risk': 50,
        'Merchant Risk': 30,
        'Bank Risk': 30,
        'Device Risk': 30,
        'Location Risk': 30,
        'Weekend Activity': 20,
        'Time Risk': 20,
        'Network Risk': 40,
        'Transaction Velocity': 40,
        'Historical Pattern': 40,
        'AML Score': tx_dict.get('probability', 0.5) * 100
    }
    
    # Create spider chart
    spider_chart = viz_engine.create_spider_chart(
        spider_features,
        f"Risk Analysis - Transaction {transaction_id}"
    )
    
    return render_template('transaction_detail.html',
                          transaction=tx_dict,
                          spider_chart=spider_chart)


@analysis_bp.route('/search')
@login_required
def search():
    """Search transactions"""
    query = request.args.get('q', '')
    field = request.args.get('field', 'all')
    
    if query:
        results = db.search_transactions(query, field)
    else:
        results = []
    
    return render_template('search.html', results=results, query=query, field=field)


@analysis_bp.route('/network-analysis')
@login_required
def network_analysis():
    """Network analysis page"""
    transactions = db.get_transactions(limit=100)
    network_data = viz_engine.create_network_graph(pd.DataFrame([dict(t) for t in transactions]))
    
    return render_template('network_analysis.html', network_data=network_data)


@analysis_bp.route('/model-comparison')
@login_required
def model_comparison():
    """Model comparison page"""
    return render_template('model_comparison.html',
                          model_status=prediction_engine.get_model_status())


@analysis_bp.route('/api/model-comparison', methods=['POST'])
@login_required
def api_model_comparison():
    """API endpoint for model comparison"""
    upload_id = request.json.get('upload_id')
    
    if not upload_id:
        return jsonify({'error': 'Upload ID required'}), 400
    
    # Get transactions from upload
    transactions = db.get_transactions(filters={'upload_id': upload_id})
    
    if not transactions:
        return jsonify({'error': 'No transactions found'}), 404
    
    # Prepare data for comparison
    df = pd.DataFrame([dict(t) for t in transactions])
    
    # Compare models
    comparison = model_comparator.compare_all_models(df)
    
    return jsonify(comparison)


@analysis_bp.route('/api/spider-chart', methods=['POST'])
@login_required
def api_spider_chart():
    """Generate spider chart for transaction"""
    transaction_id = request.json.get('transaction_id')
    
    transaction = db.get_transaction(transaction_id)
    if not transaction:
        return jsonify({'error': 'Transaction not found'}), 404
    
    tx_dict = dict(transaction)
    
    spider_features = {
        'Transaction Amount': tx_dict.get('risk_score', 50),
        'Frequency': 50,
        'Sender Risk': 50,
        'Receiver Risk': 50,
        'Merchant Risk': 30,
        'Bank Risk': 30,
        'Device Risk': 30,
        'Location Risk': 30,
        'Weekend Activity': 20,
        'Time Risk': 20,
        'Network Risk': 40,
        'Transaction Velocity': 40,
        'Historical Pattern': 40,
        'AML Score': tx_dict.get('probability', 0.5) * 100
    }
    
    spider_chart = viz_engine.create_spider_chart(spider_features)
    
    return jsonify(spider_chart)


# ==================== Report Routes ====================

@report_bp.route('/reports')
@login_required
def reports():
    """Reports page"""
    return render_template('reports.html')


@report_bp.route('/reports/generate', methods=['POST'])
@login_required
def generate_report():
    """Generate report"""
    report_type = request.form.get('report_type')
    upload_id = request.form.get('upload_id')
    
    transactions = db.get_transactions(limit=1000)
    
    if report_type == 'csv':
        df = pd.DataFrame([dict(t) for t in transactions])
        report_path = os.path.join(Config.REPORTS_FOLDER, f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        df.to_csv(report_path, index=False)
        
    elif report_type == 'excel':
        df = pd.DataFrame([dict(t) for t in transactions])
        report_path = os.path.join(Config.REPORTS_FOLDER, f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
        df.to_excel(report_path, index=False)
    
    return jsonify({'status': 'success', 'report_path': report_path})


@report_bp.route('/reports/download/<filename>')
@login_required
def download_report(filename):
    """Download report"""
    return send_file(os.path.join(Config.REPORTS_FOLDER, filename), as_attachment=True)


# ==================== Admin Routes ====================

@admin_bp.route('/admin')
@login_required
def admin():
    """Admin page"""
    if session.get('role') != 'admin':
        return render_template('error.html', error='Access denied'), 403
    
    model_performance = db.get_model_performance()
    system_logs = db.get_system_logs()
    
    return render_template('admin.html',
                          model_performance=model_performance,
                          system_logs=system_logs)


@admin_bp.route('/api/system-logs')
@login_required
def api_system_logs():
    """API endpoint for system logs"""
    logs = db.get_system_logs()
    return jsonify([dict(log) for log in logs])


@admin_bp.route('/api/model-performance')
@login_required
def api_model_performance():
    """API endpoint for model performance"""
    performance = db.get_model_performance()
    return jsonify([dict(p) for p in performance])


# ==================== API Routes ====================

@main_bp.route('/api/models/status')
@login_required
def api_models_status():
    """Get model status"""
    return jsonify(prediction_engine.get_model_status())


@main_bp.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """Make prediction on single transaction"""
    data = request.json
    model_name = data.get('model', 'random_forest')
    
    # Prepare features
    features = data.get('features', {})
    
    try:
        result = prediction_engine.predict_single(features, model_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
