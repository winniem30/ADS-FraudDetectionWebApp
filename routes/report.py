"""
Report Blueprint
Handles report generation (CSV, Excel, PDF)
"""

from flask import Blueprint, render_template, request, session, send_file, jsonify
from database import db
import pandas as pd
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

report_bp = Blueprint('report', __name__, url_prefix='/report')


@report_bp.route('/reports')
def reports():
    """Reports page"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    try:
        # Get available uploads for report generation
        from database import db
        uploads = []
        # This would need to be implemented in database.py
        # For now, return empty list
        
        return render_template('reports.html', uploads=uploads)
    except Exception as e:
        logger.error(f"Reports error: {str(e)}")
        return render_template('error.html', error=str(e))


@report_bp.route('/generate', methods=['POST'])
def generate_report():
    """Generate report in specified format"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = request.get_json()
        upload_id = data.get('upload_id')
        report_format = data.get('format', 'csv')
        include_charts = data.get('include_charts', True)
        
        # Get transactions for upload
        # This would need to be implemented in database.py
        transactions = []
        
        # Create DataFrame
        df = pd.DataFrame(transactions)
        
        # Generate report based on format
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if report_format == 'csv':
            filename = f"report_{timestamp}.csv"
            filepath = os.path.join('reports', filename)
            df.to_csv(filepath, index=False)
            return jsonify({'success': True, 'filename': filename})
        
        elif report_format == 'excel':
            filename = f"report_{timestamp}.xlsx"
            filepath = os.path.join('reports', filename)
            with pd.ExcelWriter(filepath) as writer:
                df.to_excel(writer, sheet_name='Transactions', index=False)
                # Add summary sheet if needed
            return jsonify({'success': True, 'filename': filename})
        
        elif report_format == 'pdf':
            filename = f"report_{timestamp}.pdf"
            filepath = os.path.join('reports', filename)
            # Generate PDF using reportlab or similar
            # For now, return placeholder
            return jsonify({'success': True, 'filename': filename})
        
        else:
            return jsonify({'error': 'Invalid format'}), 400
        
    except Exception as e:
        logger.error(f"Generate report error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@report_bp.route('/download/<filename>')
def download_report(filename):
    """Download generated report"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    try:
        filepath = os.path.join('reports', filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return render_template('error.html', error='Report not found')
    except Exception as e:
        logger.error(f"Download report error: {str(e)}")
        return render_template('error.html', error=str(e))
