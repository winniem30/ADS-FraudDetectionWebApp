"""
Routes package for AML Detection Platform
Contains all Flask blueprints for modular routing
"""

from .auth import auth_bp
from .dashboard import dashboard_bp
from .upload import upload_bp
from .analysis import analysis_bp
from .report import report_bp
from .admin import admin_bp

__all__ = ['auth_bp', 'dashboard_bp', 'upload_bp', 'analysis_bp', 'report_bp', 'admin_bp']
