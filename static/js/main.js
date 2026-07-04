/**
 * AML Detection Platform - Main JavaScript
 * Handles global functionality, theme toggling, and interactive features
 */

// ============================================
// Theme Toggle
// ============================================

const themeToggle = document.getElementById('theme-toggle');
if (themeToggle) {
    themeToggle.addEventListener('click', function() {
        document.body.classList.toggle('dark-mode');
        
        const isDark = document.body.classList.contains('dark-mode');
        const icon = this.querySelector('i');
        
        if (isDark) {
            icon.classList.remove('fa-moon');
            icon.classList.add('fa-sun');
            localStorage.setItem('theme', 'dark');
        } else {
            icon.classList.remove('fa-sun');
            icon.classList.add('fa-moon');
            localStorage.setItem('theme', 'light');
        }
    });
    
    // Load saved theme
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-mode');
        const icon = themeToggle.querySelector('i');
        icon.classList.remove('fa-moon');
        icon.classList.add('fa-sun');
    }
}

// ============================================
// Alerts
// ============================================

const alertsBtn = document.getElementById('alerts-btn');
const alertsCount = document.getElementById('alerts-count');

if (alertsBtn && alertsCount) {
    // Fetch unread alerts count
    fetch('/dashboard/api/alerts')
        .then(response => response.json())
        .then(data => {
            const unreadCount = data.filter(alert => !alert.is_read).length;
            alertsCount.textContent = unreadCount;
            alertsCount.style.display = unreadCount > 0 ? 'inline' : 'none';
        })
        .catch(error => {
            console.error('Error fetching alerts:', error);
        });
    
    // Show alerts dropdown on click
    alertsBtn.addEventListener('click', function() {
        // Could implement a dropdown here
        alert('Alerts feature - implement dropdown here');
    });
}

// ============================================
// Sidebar Toggle (Mobile)
// ============================================

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    if (sidebar) {
        sidebar.classList.toggle('open');
    }
}

// ============================================
// Loading Spinner
// ============================================

function showLoading() {
    const spinner = document.createElement('div');
    spinner.className = 'spinner';
    spinner.id = 'globalSpinner';
    document.body.appendChild(spinner);
}

function hideLoading() {
    const spinner = document.getElementById('globalSpinner');
    if (spinner) {
        spinner.remove();
    }
}

// ============================================
// Toast Notifications
// ============================================

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <div class="toast-content">
            <i class="fas ${getToastIcon(type)}"></i>
            <span>${message}</span>
        </div>
        <button class="toast-close">&times;</button>
    `;
    
    document.body.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 300);
    }, 5000);
    
    // Close button
    toast.querySelector('.toast-close').addEventListener('click', () => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 300);
    });
}

function getToastIcon(type) {
    switch(type) {
        case 'success':
            return 'fa-check-circle';
        case 'error':
            return 'fa-exclamation-circle';
        case 'warning':
            return 'fa-exclamation-triangle';
        default:
            return 'fa-info-circle';
    }
}

// ============================================
// Confirm Dialog
// ============================================

function confirmDialog(message, callback) {
    if (confirm(message)) {
        callback();
    }
}

// ============================================
// Format Functions
// ============================================

function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

function formatNumber(num) {
    return new Intl.NumberFormat('en-US').format(num);
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// ============================================
// API Helper
// ============================================

async function apiRequest(url, options = {}) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };
    
    const mergedOptions = { ...defaultOptions, ...options };
    
    try {
        const response = await fetch(url, mergedOptions);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        showToast('An error occurred. Please try again.', 'error');
        throw error;
    }
}

// ============================================
// Chart Helper
// ============================================

function createChart(canvasId, type, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: true,
                position: 'bottom'
            }
        }
    };
    
    const mergedOptions = { ...defaultOptions, ...options };
    
    return new Chart(ctx, {
        type: type,
        data: data,
        options: mergedOptions
    });
}

// ============================================
// Table Helper
// ============================================

function sortTable(table, column, direction = 'asc') {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    rows.sort((a, b) => {
        const aVal = a.cells[column].textContent.trim();
        const bVal = b.cells[column].textContent.trim();
        
        // Try numeric comparison
        const aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ''));
        const bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ''));
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return direction === 'asc' ? aNum - bNum : bNum - aNum;
        }
        
        // String comparison
        return direction === 'asc' 
            ? aVal.localeCompare(bVal) 
            : bVal.localeCompare(aVal);
    });
    
    rows.forEach(row => tbody.appendChild(row));
}

// ============================================
// Export Functions
// ============================================

function exportToCSV(data, filename) {
    const csv = data.map(row => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
}

function exportToJSON(data, filename) {
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
}

// ============================================
// Validation
// ============================================

function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

function validateRequired(value) {
    return value && value.trim() !== '';
}

// ============================================
// Debounce
// ============================================

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ============================================
// Prediction Page Functions
// ============================================

function loadPredictionDetails(transactionId) {
    fetch(`/analysis/transaction/${transactionId}`)
        .then(response => response.json())
        .then(data => {
            // Update prediction UI
            updatePredictionUI(data);
        })
        .catch(error => {
            console.error('Error loading prediction:', error);
            showToast('Error loading prediction details', 'error');
        });
}

function updatePredictionUI(data) {
    // Update prediction badge
    const badge = document.querySelector('.prediction-badge');
    if (badge) {
        badge.className = `prediction-badge ${data.prediction === 1 ? 'fraud' : 'legitimate'}`;
        badge.textContent = data.prediction === 1 ? 'FRAUD DETECTED' : 'LEGITIMATE';
    }
    
    // Update details
    const details = document.querySelectorAll('.detail-item .value');
    if (details.length >= 6) {
        details[0].textContent = `${(data.probability * 100).toFixed(2)}%`;
        details[1].textContent = `${data.risk_score.toFixed(2)}/100`;
        details[2].textContent = data.risk_level.toUpperCase();
        details[2].className = `value risk-${data.risk_level}`;
        details[3].textContent = `${data.confidence_score.toFixed(2)}%`;
        details[4].textContent = data.model_used.toUpperCase();
        details[5].textContent = `${data.execution_time.toFixed(4)}s`;
    }
}

// ============================================
// Spider Chart Functions
// ============================================

function loadSpiderChart(transactionId, datasetType) {
    fetch(`/analysis/transaction/${transactionId}/spider-chart?dataset_type=${datasetType}`)
        .then(response => response.json())
        .then(data => {
            // Update spider chart
            const iframe = document.querySelector('#spider-chart-container iframe');
            if (iframe && data.html) {
                iframe.src = data.html;
            }
            
            // Update axis details
            updateAxisDetails(data.spider_features, data.axes);
        })
        .catch(error => {
            console.error('Error loading spider chart:', error);
        });
}

function updateAxisDetails(features, axes) {
    const axisList = document.querySelector('.axis-list');
    if (axisList) {
        axisList.innerHTML = axes.map(axis => `
            <div class="axis-item">
                <span class="axis-name">${axis}</span>
                <span class="axis-value">${features.get(axis, 50)}/100</span>
            </div>
        `).join('');
    }
}

function exportChart(format) {
    const chartPath = document.querySelector('#spider-chart-container iframe')?.src;
    if (chartPath) {
        const exportPath = chartPath.replace('.html', format === 'png' ? '.png' : '.html');
        window.open(exportPath, '_blank');
    }
}

// ============================================
// Analytics Page Functions
// ============================================

function loadAnalyticsData(timeRange) {
    fetch(`/dashboard/api/analytics?range=${timeRange}`)
        .then(response => response.json())
        .then(data => {
            updateAnalyticsCharts(data);
        })
        .catch(error => {
            console.error('Error loading analytics:', error);
        });
}

function updateAnalyticsCharts(data) {
    // Update transaction trend chart
    updateChart('transactionTrendChart', data.trend);
    
    // Update risk distribution chart
    updateChart('riskDistributionChart', data.risk_distribution);
    
    // Update fraud by category chart
    updateChart('fraudByCategoryChart', data.fraud_by_category);
    
    // Update fraud by time chart
    updateChart('fraudByTimeChart', data.fraud_by_time);
}

function updateTimeRange(range) {
    // Update active button
    document.querySelectorAll('.range-controls .btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.textContent === range) {
            btn.classList.add('active');
        }
    });
    
    // Load new data
    loadAnalyticsData(range);
}

// ============================================
// Settings Page Functions
// ============================================

function saveSettings(formId) {
    const form = document.getElementById(formId);
    const formData = new FormData(form);
    
    fetch('/settings/save', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('Settings saved successfully!', 'success');
        } else {
            showToast('Error saving settings', 'error');
        }
    })
    .catch(error => {
        console.error('Error saving settings:', error);
        showToast('Error saving settings', 'error');
    });
}

function clearCache() {
    if (confirmDialog('Are you sure you want to clear the cache? This will delete all cached charts and temporary files.')) {
        fetch('/settings/clear-cache', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showToast('Cache cleared successfully!', 'success');
                } else {
                    showToast('Error clearing cache', 'error');
                }
            })
            .catch(error => {
                console.error('Error clearing cache:', error);
                showToast('Error clearing cache', 'error');
            });
    }
}

function resetDatabase() {
    if (confirmDialog('⚠️ WARNING: This will permanently delete ALL data. This action cannot be undone. Are you sure?')) {
        fetch('/settings/reset-database', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showToast('Database reset successfully!', 'success');
                    setTimeout(() => location.reload(), 2000);
                } else {
                    showToast('Error resetting database', 'error');
                }
            })
            .catch(error => {
                console.error('Error resetting database:', error);
                showToast('Error resetting database', 'error');
            });
    }
}

// ============================================
// Model Comparison Functions
// ============================================

function compareModels(uploadId, datasetType) {
    showLoading();
    
    fetch('/analysis/api/model-comparison', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            upload_id: uploadId,
            dataset_type: datasetType
        })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        updateComparisonCharts(data);
    })
    .catch(error => {
        hideLoading();
        console.error('Error comparing models:', error);
        showToast('Error comparing models', 'error');
    });
}

function updateComparisonCharts(data) {
    // Update comparison charts
    for (const [modelName, results] of Object.entries(data)) {
        if (results.error) continue;
        
        const chartId = `comparison-${modelName}`;
        updateChart(chartId, {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            datasets: [{
                label: modelName,
                data: [results.accuracy, results.precision, results.recall, results.f1_score]
            }]
        });
    }
}

// ============================================
// Report Generation Functions
// ============================================

function generateReport(uploadId, format, includeCharts) {
    showLoading();
    
    fetch('/report/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            upload_id: uploadId,
            format: format,
            include_charts: includeCharts
        })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        if (data.success) {
            showToast(`Report generated: ${data.filename}`, 'success');
            // Trigger download
            window.location.href = `/report/download/${data.filename}`;
        } else {
            showToast('Error generating report', 'error');
        }
    })
    .catch(error => {
        hideLoading();
        console.error('Error generating report:', error);
        showToast('Error generating report', 'error');
    });
}

// ============================================
// Network Analysis Functions
// ============================================

function loadNetworkGraph(uploadId) {
    showLoading();
    
    fetch(`/analysis/network-analysis?upload_id=${uploadId}`)
        .then(response => response.json())
        .then(data => {
            hideLoading();
            renderNetworkGraph(data);
        })
        .catch(error => {
            hideLoading();
            console.error('Error loading network graph:', error);
            showToast('Error loading network graph', 'error');
        });
}

function renderNetworkGraph(data) {
    const container = document.getElementById('network-graph-container');
    if (!container) return;
    
    // Use D3.js to render network graph
    // This is a placeholder - actual implementation would use D3.js
    container.innerHTML = `
        <div class="network-placeholder">
            <i class="fas fa-project-diagram"></i>
            <p>Network graph visualization</p>
            <p>Nodes: ${data.node_count}</p>
            <p>Edges: ${data.edge_count}</p>
            <p>Suspicious paths: ${data.suspicious_paths}</p>
        </div>
    `;
}

// ============================================
// Search Functions
// ============================================

function performSearch(query, field) {
    if (!query.trim()) {
        showToast('Please enter a search term', 'warning');
        return;
    }
    
    showLoading();
    
    fetch(`/analysis/search?q=${encodeURIComponent(query)}&field=${field}`)
        .then(response => response.json())
        .then(data => {
            hideLoading();
            displaySearchResults(data);
        })
        .catch(error => {
            hideLoading();
            console.error('Error searching:', error);
            showToast('Error performing search', 'error');
        });
}

function displaySearchResults(results) {
    const resultsContainer = document.getElementById('search-results');
    if (!resultsContainer) return;
    
    if (results.length === 0) {
        resultsContainer.innerHTML = '<p class="no-results">No results found</p>';
        return;
    }
    
    resultsContainer.innerHTML = results.map(result => `
        <div class="search-result-item">
            <div class="result-header">
                <span class="transaction-id">${result.transaction_id || 'N/A'}</span>
                <span class="risk-badge risk-${result.risk_level}">${result.risk_level.toUpperCase()}</span>
            </div>
            <div class="result-details">
                <span>Amount: ${formatCurrency(result.amount || 0)}</span>
                <span>Date: ${formatDate(result.timestamp)}</span>
            </div>
            <button class="btn btn-sm btn-primary" onclick="viewTransaction(${result.id})">View Details</button>
        </div>
    `).join('');
}

function viewTransaction(transactionId) {
    window.location.href = `/analysis/transaction/${transactionId}`;
}

// ============================================
// Initialize on DOM Ready
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    // Add event listeners for settings forms
    const settingsForms = ['profileForm', 'securityForm', 'modelForm', 'notificationForm', 'displayForm'];
    settingsForms.forEach(formId => {
        const form = document.getElementById(formId);
        if (form) {
            form.addEventListener('submit', function(e) {
                e.preventDefault();
                saveSettings(formId);
            });
        }
    });
    
    // Add event listeners for time range buttons
    const rangeButtons = document.querySelectorAll('.range-controls .btn');
    rangeButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            updateTimeRange(this.textContent);
        });
    });
    
    // Add event listener for search form
    const searchForm = document.getElementById('searchForm');
    if (searchForm) {
        searchForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const query = document.getElementById('searchQuery').value;
            const field = document.getElementById('searchField').value;
            performSearch(query, field);
        });
    }
    
    // Initialize all existing functionality
    const cards = document.querySelectorAll('.stat-card, .chart-card, .alert-card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in');
    });
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
});

// ============================================
// Error Handler
// ============================================

window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
    showToast('An unexpected error occurred', 'error');
});

// ============================================
// Keyboard Shortcuts
// ============================================

document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K for search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.querySelector('input[type="search"], input[placeholder*="search" i]');
        if (searchInput) {
            searchInput.focus();
        }
    }
    
    // Escape to close modals
    if (e.key === 'Escape') {
        const modals = document.querySelectorAll('.modal.show');
        modals.forEach(modal => {
            const bsModal = bootstrap.Modal.getInstance(modal);
            if (bsModal) {
                bsModal.hide();
            }
        });
    }
});

// ============================================
// Print Function
// ============================================

function printPage() {
    window.print();
}

// ============================================
// Copy to Clipboard
// ============================================

async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showToast('Copied to clipboard!', 'success');
    } catch (err) {
        console.error('Failed to copy:', err);
        showToast('Failed to copy to clipboard', 'error');
    }
}

// ============================================
// Local Storage Helper
// ============================================

const storage = {
    get(key) {
        const item = localStorage.getItem(key);
        return item ? JSON.parse(item) : null;
    },
    set(key, value) {
        localStorage.setItem(key, JSON.stringify(value));
    },
    remove(key) {
        localStorage.removeItem(key);
    },
    clear() {
        localStorage.clear();
    }
};
