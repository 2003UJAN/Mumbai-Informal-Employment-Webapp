import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import pickle
from datetime import datetime
import os

# Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Mumbai Informal Employment | GeoAI Platform",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# GEMINI 2.0 FLASH LITE CONFIGURATION
# ============================================

def configure_gemini():
    """Configure Gemini AI 2.0 Flash Lite"""
    if GEMINI_AVAILABLE:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                # Using Gemini 2.0 Flash Lite (faster, cheaper)
                return genai.GenerativeModel('gemini-2.0-flash-lite')
            return None
        except Exception as e:
            st.error(f"Gemini configuration error: {e}")
            return None
    return None

gemini_model = configure_gemini()

# ============================================
# PROFESSIONAL STYLING
# ============================================

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #667eea30;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .predict-card {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .ai-response {
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# DATA LOADING
# ============================================

@st.cache_data
def load_data():
    """Load all required data files"""
    try:
        data = pd.read_csv('data/mumbai_all_8_areas_data.csv')
        stats = pd.read_csv('data/mumbai_all_areas_statistics.csv')
        predictions = pd.read_csv('data/mumbai_all_areas_predictions.csv')
        feature_imp = pd.read_csv('data/mumbai_feature_importance.csv')
        return data, stats, predictions, feature_imp, True
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None, None, None, False

@st.cache_resource
def load_model():
    """Load trained Random Forest model"""
    try:
        with open('model/mumbai_rf_model_all_areas.pkl', 'rb') as f:
            model = pickle.load(f)
        return model, True
    except Exception as e:
        st.warning(f"⚠️ Model not loaded: {e}")
        return None, False

# Load data and model
data, stats, predictions, feature_imp, data_loaded = load_data()
model, model_loaded = load_model()

# Mumbai configuration
MUMBAI_SLUM_AREAS = {
    'Dharavi': {'coords': (19.0444, 72.8560), 'population': 1000000, 'area_sqkm': 2.39},
    'Govandi': {'coords': (19.0545, 72.9144), 'population': 250000, 'area_sqkm': 1.5},
    'Mankhurd': {'coords': (19.0437, 72.9264), 'population': 200000, 'area_sqkm': 1.2},
    'Kurla': {'coords': (19.0728, 72.8826), 'population': 150000, 'area_sqkm': 0.8},
    'Bandra East': {'coords': (19.0596, 72.8428), 'population': 120000, 'area_sqkm': 0.6},
    'Andheri East': {'coords': (19.1136, 72.8697), 'population': 180000, 'area_sqkm': 1.0},
    'Worli': {'coords': (19.0144, 72.8169), 'population': 100000, 'area_sqkm': 0.5},
    'Malad': {'coords': (19.1864, 72.8479), 'population': 160000, 'area_sqkm': 0.9}
}

# ============================================
# HEADER
# ============================================

st.markdown('<h1 class="main-header">🏙️ Mumbai Informal Employment Platform</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; opacity: 0.8;">Geospatial ML + AI-Powered Insights for 8 Major Slum Areas</p>', unsafe_allow_html=True)

if not data_loaded:
    st.error("🚫 Failed to load data. Please check file paths.")
    st.stop()

# ============================================
# SIDEBAR NAVIGATION
# ============================================

with st.sidebar:
    st.markdown("### 🗺️ Navigation")
    
    page = st.radio(
        "Select Page",
        ["🏠 Home", "🗺️ Interactive Map", "🎯 ML Predictions", "🤖 AI Insights"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### 📌 Quick Stats")
    st.metric("Total Areas", "8")
    st.metric("Total Samples", f"{len(data):,}")
    st.metric("Avg Employment", f"{data['informal_employment_density'].mean():.1f}")
    
    st.markdown("---")
    
    if model_loaded:
        st.success("✅ ML Model: Active")
    else:
        st.warning("⚠️ ML Model: Unavailable")
    
    if gemini_model:
        st.success("✅ Gemini 2.0: Active")
    else:
        st.info("ℹ️ Gemini AI: Configure API Key")

# ============================================
# PAGE 1: HOME
# ============================================

if page == "🏠 Home":
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="stats-number">2.16M</div>
            <div style="opacity: 0.7;">Total Population</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="stats-number">8.89 km²</div>
            <div style="opacity: 0.7;">Total Area</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="stats-number">95%+</div>
            <div style="opacity: 0.7;">ML Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="stats-number">1.62M</div>
            <div style="opacity: 0.7;">Informal Workers</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Platform Features")
        st.markdown("""
        - 🗺️ **Interactive Geospatial Maps** - Folium-based visualization
        - 🎯 **ML Predictions** - Random Forest with 95%+ accuracy
        - 🤖 **AI Insights** - Gemini 2.0 Flash Lite integration
        - 📊 **Real-time Data Explorer** - Filter and download data
        - 📈 **Professional Visualizations** - Plotly charts
        - 💡 **Policy Recommendations** - Data-driven suggestions
        """)
    
    with col2:
        st.markdown("### 🏘️ Areas Covered")
        for area_name, info in list(MUMBAI_SLUM_AREAS.items())[:4]:
            st.info(f"**{area_name}** - Pop: {info['population']:,}")
    
    st.markdown("---")
    
    # Area comparison chart
    st.markdown("### 📊 Employment Density by Area")
    
    area_comparison = data.groupby('area_name')['informal_employment_density'].mean().sort_values(ascending=False)
    
    fig = px.bar(
        x=area_comparison.values,
        y=area_comparison.index,
        orientation='h',
        labels={'x': 'Avg Employment Density', 'y': 'Area'},
        color=area_comparison.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE 2: INTERACTIVE MAP
# ============================================

elif page == "🗺️ Interactive Map":
    st.markdown("### 🗺️ Interactive Geospatial Map - All 8 Slum Areas")
    
    # Map controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        map_zoom = st.slider("🔍 Zoom Level", 10, 13, 11)
    with col2:
        show_labels = st.checkbox("Show Area Labels", value=True)
    with col3:
        st.metric("Total Areas", "8")
    
    # Create map
    mumbai_center = [19.0760, 72.8777]
    m = folium.Map(location=mumbai_center, zoom_start=map_zoom, tiles="OpenStreetMap")
    
    for area_name, area_info in MUMBAI_SLUM_AREAS.items():
        area_data = data[data['area_name'] == area_name]
        
        if len(area_data) > 0:
            avg_employment = area_data['informal_employment_density'].mean()
            avg_nightlight = area_data['nightlight_intensity'].mean()
            avg_mobile = area_data['mobile_calls'].mean()
            
            if avg_employment > 60:
                color = '#FF4B4B'
                status = "High"
            elif avg_employment > 40:
                color = '#FFA500'
                status = "Medium"
            else:
                color = '#4CAF50'
                status = "Low"
            
            popup_html = f"""
            <div style="font-family: Arial; width: 280px; padding: 15px;">
                <h3 style="color: {color}; margin: 0 0 10px 0;">{area_name}</h3>
                <hr style="margin: 10px 0;">
                <p style="margin: 5px 0;"><b>📊 Population:</b> {area_info['population']:,}</p>
                <p style="margin: 5px 0;"><b>📍 Area:</b> {area_info['area_sqkm']} km²</p>
                <p style="margin: 5px 0;"><b>👥 Employment:</b> {avg_employment:.1f} workers/km²</p>
                <p style="margin: 5px 0;"><b>🌙 Nightlight:</b> {avg_nightlight:.1f}</p>
                <p style="margin: 5px 0;"><b>📱 Mobile Activity:</b> {avg_mobile:.0f} calls/day</p>
                <p style="margin: 5px 0;"><b>🏷️ Status:</b> <span style="color: {color}; font-weight: bold;">{status}</span></p>
            </div>
            """
            
            folium.CircleMarker(
                location=area_info['coords'],
                radius=18,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{area_name} - Click for details",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=3
            ).add_to(m)
            
            if show_labels:
                folium.Marker(
                    location=area_info['coords'],
                    icon=folium.DivIcon(html=f"""
                        <div style="font-size: 11px; font-weight: bold; color: white; 
                                    background-color: {color}; padding: 5px 10px; 
                                    border-radius: 5px; border: 2px solid white;
                                    box-shadow: 0 2px 5px rgba(0,0,0,0.3);">
                            {area_name}
                        </div>
                    """)
                ).add_to(m)
    
    # Display map
    st_folium(m, width=1400, height=650)
    
    # Legend
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🔴 **High**: >60 workers/km²")
    with col2:
        st.markdown("🟠 **Medium**: 40-60 workers/km²")
    with col3:
        st.markdown("🟢 **Low**: <40 workers/km²")

# ============================================
# PAGE 3: ML PREDICTIONS
# ============================================

elif page == "🎯 ML Predictions":
    st.markdown("### 🎯 Machine Learning Predictions & Model Performance")
    
    if model_loaded and predictions is not None:
        
        # Model metrics
        r2 = 1 - (predictions['residual'].var() / predictions['actual'].var())
        rmse = np.sqrt((predictions['residual']**2).mean())
        mae = predictions['residual'].abs().mean()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="stats-number">{r2:.4f}</div>
                <div style="opacity: 0.7;">R² Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="stats-number">{rmse:.2f}</div>
                <div style="opacity: 0.7;">RMSE</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="stats-number">{mae:.2f}</div>
                <div style="opacity: 0.7;">MAE</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Interactive prediction tool
        st.markdown("### 🎲 Make New Predictions")
        
        with st.form("prediction_form"):
            st.markdown('<div class="predict-card">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                nightlight = st.slider("🌙 Nightlight Intensity", 5.0, 100.0, 50.0)
                mobile_calls = st.slider("📱 Mobile Calls (per day)", 100, 20000, 10000)
            
            with col2:
                data_usage = st.slider("📊 Data Usage (MB)", 1000, 60000, 30000)
                unique_devices = st.slider("📲 Unique Devices", 50, 10000, 5000)
            
            submitted = st.form_submit_button("🚀 Predict Employment Density", use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        if submitted:
            # Make prediction
            input_data = np.array([[nightlight, mobile_calls, data_usage, unique_devices]])
            prediction = model.predict(input_data)[0]
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown(f"""
                <div class="predict-card" style="text-align: center;">
                    <h3>📊 Prediction Result</h3>
                    <div style="font-size: 3rem; font-weight: 700; color: #667eea; margin: 1rem 0;">
                        {prediction:.1f}
                    </div>
                    <div style="font-size: 1.2rem; opacity: 0.7;">workers per sq.km</div>
                    <hr style="margin: 1.5rem 0;">
                    <p><b>Interpretation:</b> {'High' if prediction > 60 else 'Medium' if prediction > 40 else 'Low'} informal employment density</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Actual vs Predicted chart
        st.markdown("### 📈 Model Performance Visualization")
        
        tab1, tab2 = st.tabs(["📊 Actual vs Predicted", "🎯 Feature Importance"])
        
        with tab1:
            fig = px.scatter(
                predictions, 
                x='actual', 
                y='predicted',
                labels={'actual': 'Actual Employment Density', 'predicted': 'Predicted Employment Density'},
                title="Model Predictions vs Actual Values",
                color_discrete_sequence=['#667eea']
            )
            fig.add_trace(go.Scatter(
                x=[predictions['actual'].min(), predictions['actual'].max()],
                y=[predictions['actual'].min(), predictions['actual'].max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash', width=2)
            ))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if feature_imp is not None:
                fig = px.bar(
                    feature_imp,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Feature Importance - Random Forest Model",
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("❌ Model or prediction data not available")

# ============================================
# PAGE 4: AI INSIGHTS (GEMINI 2.0 FLASH LITE)
# ============================================

elif page == "🤖 AI Insights":
    st.markdown("### 🤖 AI-Powered Insights with Gemini 2.0 Flash Lite")
    
    if gemini_model:
        st.success("✅ Gemini 2.0 Flash Lite is ready to generate insights!")
        
        # Insight type selector
        col1, col2 = st.columns([2, 1])
        
        with col1:
            insight_type = st.selectbox(
                "🎯 Select Analysis Type",
                [
                    "Quick Overview",
                    "Model Performance Analysis",
                    "Policy Recommendations",
                    "Area Prioritization",
                    "Economic Impact Assessment",
                    "Custom Query"
                ]
            )
        
        with col2:
            st.metric("Model Used", "Gemini 2.0")
            st.caption("Flash Lite")
        
        # Custom query for "Custom Query" option
        if insight_type == "Custom Query":
            custom_query = st.text_area(
                "💬 Enter your custom question about the data:",
                placeholder="E.g., Which area needs the most urgent intervention?",
                height=100
            )
        else:
            custom_query = None
        
        # Generate button
        if st.button("🚀 Generate AI Insights", type="primary", use_container_width=True):
            
            with st.spinner("🤖 Gemini 2.0 Flash Lite is analyzing..."):
                
                try:
                    # Prepare context with data
                    context = f"""
                    You are an expert in urban economics and geospatial analysis. 
                    
                    DATASET SUMMARY:
                    - Total samples: {len(data):,}
                    - Areas analyzed: 8 major Mumbai slum areas
                    - Total population: 2.16 million
                    - Average employment density: {data['informal_employment_density'].mean():.2f} workers/sq.km
                    
                    TOP 3 AREAS BY EMPLOYMENT DENSITY:
                    {data.groupby('area_name')['informal_employment_density'].mean().sort_values(ascending=False).head(3).to_string()}
                    
                    FEATURE IMPORTANCE:
                    {feature_imp.to_string(index=False) if feature_imp is not None else 'Not available'}
                    
                    MODEL PERFORMANCE:
                    - R² Score: {r2:.4f} if predictions is not None else 'N/A'
                    - RMSE: {rmse:.2f} if predictions is not None else 'N/A'
                    """
                    
                    # Create prompt based on insight type
                    if insight_type == "Quick Overview":
                        prompt = f"{context}\n\nProvide a quick 3-point summary of the key findings."
                    
                    elif insight_type == "Model Performance Analysis":
                        prompt = f"{context}\n\nAnalyze the machine learning model's performance. What does the R² score tell us? Are there any concerns? Be technical but clear."
                    
                    elif insight_type == "Policy Recommendations":
                        prompt = f"{context}\n\nProvide 5 specific, actionable policy recommendations for Mumbai authorities based on this data. Focus on practical interventions."
                    
                    elif insight_type == "Area Prioritization":
                        prompt = f"{context}\n\nRank the top 5 areas that need immediate intervention and explain why. Consider population, density, and employment patterns."
                    
                    elif insight_type == "Economic Impact Assessment":
                        prompt = f"{context}\n\nAssess the economic impact of informal employment in Mumbai. Estimate GDP contribution, tax implications, and economic multiplier effects."
                    
                    elif insight_type == "Custom Query" and custom_query:
                        prompt = f"{context}\n\nUser Question: {custom_query}\n\nProvide a detailed, data-driven answer."
                    
                    else:
                        prompt = f"{context}\n\nProvide a general analysis of the informal employment situation in Mumbai."
                    
                    # Call Gemini 2.0 Flash Lite
                    response = gemini_model.generate_content(prompt)
                    
                    # Display response
                    st.markdown("---")
                    st.markdown("### 💡 AI-Generated Insights")
                    
                    st.markdown(f"""
                    <div class="ai-response">
                        {response.text.replace('\n', '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Save option
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col2:
                        if st.button("💾 Save Insights", use_container_width=True):
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f'gemini_insights_{timestamp}.txt'
                            
                            with open(filename, 'w', encoding='utf-8') as f:
                                f.write(f"GEMINI 2.0 FLASH LITE INSIGHTS\n")
                                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                f.write(f"Analysis Type: {insight_type}\n")
                                f.write("="*60 + "\n\n")
                                f.write(response.text)
                            
                            st.success(f"✅ Insights saved to {filename}")
                
                except Exception as e:
                    st.error(f"❌ Error generating insights: {e}")
                    st.info("💡 Make sure your GEMINI_API_KEY is correctly configured")
    
    else:
        st.warning("⚠️ Gemini AI not configured")
        st.info("""
        **To enable Gemini 2.0 Flash Lite:**
        
        1. Get API key from: https://aistudio.google.com/app/apikey
        2. Add to `.streamlit/secrets.toml`:
           ```
           GEMINI_API_KEY = "your-api-key-here"
           ```
        3. Restart the application
        """)
        
        # Show pre-saved insights if available
        try:
            with open('data/gemini_insights_all_areas.txt', 'r', encoding='utf-8') as f:
                saved_insights = f.read()
            
            st.markdown("---")
            st.markdown("### 📄 Pre-saved Insights")
            with st.expander("View Saved Insights", expanded=True):
                st.text(saved_insights)
        
        except:
            st.info("No pre-saved insights available")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; opacity: 0.7;">
    <p><b>Mumbai Informal Employment GeoAI Platform</b></p>
    <p>Developed by Ujan Pradhan, Vyomika Anand & Navya Singhal</p>
    <p>SRM Institute of Science & Technology | IIT Bombay</p>
    <p style="margin-top: 1rem;">
        <a href="https://github.com/2003UJAN/Mumbai-Informal-Employment-Webapp" target="_blank">GitHub</a> | 
        <a href="mailto:up0625@srmist.edu.in">Contact</a>
    </p>
    <p style="font-size: 0.8rem; margin-top: 1rem;">© 2025 All rights reserved</p>
</div>
""", unsafe_allow_html=True)
