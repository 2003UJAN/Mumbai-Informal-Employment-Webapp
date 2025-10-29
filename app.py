from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime

# Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Mumbai configuration
MUMBAI_SLUM_AREAS = {
    'Dharavi': {'coords': [19.0444, 72.8560], 'population': 1000000, 'area_sqkm': 2.39},
    'Govandi': {'coords': [19.0545, 72.9144], 'population': 250000, 'area_sqkm': 1.5},
    'Mankhurd': {'coords': [19.0437, 72.9264], 'population': 200000, 'area_sqkm': 1.2},
    'Kurla': {'coords': [19.0728, 72.8826], 'population': 150000, 'area_sqkm': 0.8},
    'Bandra East': {'coords': [19.0596, 72.8428], 'population': 120000, 'area_sqkm': 0.6},
    'Andheri East': {'coords': [19.1136, 72.8697], 'population': 180000, 'area_sqkm': 1.0},
    'Worli': {'coords': [19.0144, 72.8169], 'population': 100000, 'area_sqkm': 0.5},
    'Malad': {'coords': [19.1864, 72.8479], 'population': 160000, 'area_sqkm': 0.9}
}

# Load data on startup
try:
    data = pd.read_csv('data/mumbai_all_8_areas_data.csv')
    stats = pd.read_csv('data/mumbai_all_areas_statistics.csv')
    predictions = pd.read_csv('data/mumbai_all_areas_predictions.csv')
    feature_imp = pd.read_csv('data/mumbai_feature_importance.csv')
    DATA_LOADED = True
except Exception as e:
    print(f"Error loading data: {e}")
    DATA_LOADED = False
    data = None

# Load model
try:
    with open('model/mumbai_rf_model_all_areas.pkl', 'rb') as f:
        model = pickle.load(f)
    MODEL_LOADED = True
except:
    MODEL_LOADED = False
    model = None

# Configure Gemini AI
if GEMINI_AVAILABLE:
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            gemini_model = None
    except:
        gemini_model = None
else:
    gemini_model = None

# Routes
@app.route('/')
def index():
    """Home page"""
    if not DATA_LOADED:
        return render_template('error.html', message="Data not loaded")
    
    stats_data = {
        'total_samples': len(data),
        'total_areas': len(data['area_name'].unique()),
        'avg_employment': round(data['informal_employment_density'].mean(), 1),
        'total_population': sum([area['population'] for area in MUMBAI_SLUM_AREAS.values()]),
        'total_area': sum([area['area_sqkm'] for area in MUMBAI_SLUM_AREAS.values()])
    }
    
    return render_template('index.html', 
                         stats=stats_data, 
                         areas=MUMBAI_SLUM_AREAS,
                         model_loaded=MODEL_LOADED,
                         gemini_available=gemini_model is not None)

@app.route('/data-explorer')
def data_explorer():
    """Data explorer page"""
    if not DATA_LOADED:
        return render_template('error.html', message="Data not loaded")
    
    areas = sorted(data['area_name'].unique().tolist())
    return render_template('data_explorer.html', areas=areas)

@app.route('/api/data')
def get_data():
    """API endpoint for data"""
    if not DATA_LOADED:
        return jsonify({'error': 'Data not loaded'}), 500
    
    selected_areas = request.args.get('areas', '').split(',')
    
    if selected_areas and selected_areas[0]:
        filtered_data = data[data['area_name'].isin(selected_areas)]
    else:
        filtered_data = data
    
    # Sample for performance
    sample_data = filtered_data.sample(min(1000, len(filtered_data)))
    
    return jsonify({
        'data': sample_data.to_dict('records'),
        'total_samples': len(filtered_data)
    })

@app.route('/map')
def map_view():
    """Interactive map page"""
    if not DATA_LOADED:
        return render_template('error.html', message="Data not loaded")
    
    # Calculate statistics for each area
    area_stats = {}
    for area_name in MUMBAI_SLUM_AREAS.keys():
        area_data = data[data['area_name'] == area_name]
        if len(area_data) > 0:
            area_stats[area_name] = {
                'avg_employment': round(area_data['informal_employment_density'].mean(), 1),
                'avg_nightlight': round(area_data['nightlight_intensity'].mean(), 1),
                'avg_mobile': int(area_data['mobile_calls'].mean()),
                'samples': len(area_data)
            }
    
    return render_template('map.html', 
                         areas=MUMBAI_SLUM_AREAS, 
                         area_stats=area_stats)

@app.route('/model-performance')
def model_performance():
    """Model performance page"""
    if not DATA_LOADED or predictions is None:
        return render_template('error.html', message="Model data not loaded")
    
    # Calculate metrics
    r2 = 1 - (predictions['residual'].var() / predictions['actual'].var())
    rmse = np.sqrt((predictions['residual']**2).mean())
    mae = predictions['residual'].abs().mean()
    
    metrics = {
        'r2': round(r2, 4),
        'rmse': round(rmse, 2),
        'mae': round(mae, 2)
    }
    
    return render_template('model.html', 
                         metrics=metrics, 
                         feature_importance=feature_imp.to_dict('records') if feature_imp is not None else [])

@app.route('/area-comparison')
def area_comparison():
    """Area comparison page"""
    if not DATA_LOADED:
        return render_template('error.html', message="Data not loaded")
    
    comparison = data.groupby('area_name').agg({
        'informal_employment_density': 'mean',
        'nightlight_intensity': 'mean',
        'mobile_calls': 'mean',
        'unique_devices': 'mean'
    }).round(2).to_dict('index')
    
    return render_template('comparison.html', comparison=comparison)

@app.route('/insights')
def insights():
    """AI insights page"""
    # Load saved insights
    try:
        with open('data/gemini_insights_all_areas.txt', 'r', encoding='utf-8') as f:
            saved_insights = f.read()
    except:
        saved_insights = None
    
    return render_template('insights.html', 
                         insights=saved_insights,
                         gemini_available=gemini_model is not None)

@app.route('/api/generate-insight', methods=['POST'])
def generate_insight():
    """Generate AI insight via API"""
    if not gemini_model:
        return jsonify({'error': 'Gemini AI not configured'}), 400
    
    query = request.json.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        context = f"""
        Analyzing Mumbai informal employment data:
        - Total samples: {len(data)}
        - Areas: {len(MUMBAI_SLUM_AREAS)}
        - Avg employment: {data['informal_employment_density'].mean():.2f}
        
        User query: {query}
        """
        
        response = gemini_model.generate_content(context)
        return jsonify({'insight': response.text})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/policy')
def policy():
    """Policy recommendations page"""
    if not DATA_LOADED:
        return render_template('error.html', message="Data not loaded")
    
    top_areas = data.groupby('area_name')['informal_employment_density'].mean().sort_values(ascending=False).head(3)
    
    return render_template('policy.html', top_areas=top_areas.to_dict())

@app.route('/api/download/<data_type>')
def download_data(data_type):
    """Download data as CSV"""
    if not DATA_LOADED:
        return jsonify({'error': 'Data not loaded'}), 500
    
    if data_type == 'full':
        return send_file('data/mumbai_all_8_areas_data.csv', 
                        as_attachment=True,
                        download_name=f'mumbai_data_{datetime.now().strftime("%Y%m%d")}.csv')
    
    return jsonify({'error': 'Invalid data type'}), 400

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', message="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', message="Server error"), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
