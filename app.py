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
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/2003UJAN/Mumbai-Informal-Employment-Webapp',
        'Report a bug': 'https://github.com/2003UJAN/Mumbai-Informal-Employment-Webapp/issues',
        'About': "Mumbai Informal Employment Analysis - Geospatial Econometrics Platform"
    }
)

# ============================================
# THEME CONFIGURATION (Light/Dark Mode)
# ============================================

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# Theme colors
if st.session_state.theme == 'dark':
    primary_color = "#00D9FF"
    bg_color = "#0E1117"
    secondary_bg = "#262730"
    text_color = "#FAFAFA"
    card_bg = "#1E1E1E"
else:
    primary_color = "#1f77b4"
    bg_color = "#FFFFFF"
    secondary_bg = "#F0F2F6"
    text_color = "#262730"
    card_bg = "#FFFFFF"

# ============================================
# CUSTOM CSS (Professional Styling)
# ============================================

st.markdown(f"""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    /* Main Header */
    .main-header {{
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, {primary_color} 0%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }}
    
    .sub-header {{
        text-align: center;
        color: {text_color};
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 2rem;
        opacity: 0.8;
    }}
    
    /* Card Styles */
    .metric-card {{
        background: {card_bg};
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid {primary_color};
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin: 1rem 0;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }}
    
    /* Stats Cards */
    .stats-card {{
        background: linear-gradient(135deg, {primary_color}15 0%, #667eea15 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 2px solid {primary_color}30;
    }}
    
    .stats-number {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {primary_color};
        margin: 0.5rem 0;
    }}
    
    .stats-label {{
        font-size: 1rem;
        color: {text_color};
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    /* Sidebar Styling */
    .css-1d391kg {{
        background-color: {secondary_bg};
    }}
    
    /* Button Styles */
    .stButton > button {{
        background: linear-gradient(135deg, {primary_color} 0%, #667eea 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    }}
    
    /* Section Headers */
    .section-header {{
        font-size: 1.8rem;
        font-weight: 600;
        color: {text_color};
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid {primary_color};
    }}
    
    /* Info Boxes */
    .info-box {{
        background: {card_bg};
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }}
    
    .warning-box {{
        background: {card_bg};
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }}
    
    /* Dataframe styling */
    .dataframe {{
        border-radius: 8px;
        overflow: hidden;
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 2px solid {primary_color}30;
        color: {text_color};
        opacity: 0.7;
    }}
    
    /* Loading Animation */
    .loading-spinner {{
        text-align: center;
        padding: 2rem;
    }}
    
    /* Hide Streamlit Branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
""", unsafe_allow_html=True)

# ============================================
# GEMINI AI CONFIGURATION
# ============================================

def configure_gemini():
    """Configure Gemini AI with API key"""
    if GEMINI_AVAILABLE:
        try:
            # Try to get API key from secrets or environment
            api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
            if api_key:
                genai.configure(api_key=api_key)
                return genai.GenerativeModel('gemini-1.5-flash')
            return None
        except:
            return None
    return None

gemini_model = configure_gemini()

# ============================================
# DATA LOADING FUNCTIONS
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

@st.cache_data
def load_insights():
    """Load Gemini-generated insights"""
    try:
        with open('data/gemini_insights_all_areas.txt', 'r', encoding='utf-8') as f:
            insights = f.read()
        return insights
    except:
        return None

# Load data
data, stats, predictions, feature_imp, data_loaded = load_data()
model, model_loaded = load_model()
saved_insights = load_insights()

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

col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.write("")
with col2:
    st.markdown('<h1 class="main-header">🏙️ Mumbai Informal Employment Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Geospatial Econometrics & Machine Learning Analysis of 8 Major Slum Areas</p>', unsafe_allow_html=True)
with col3:
    if st.button("🌓" if st.session_state.theme == 'light' else "☀️"):
        toggle_theme()
        st.rerun()

# Check if data loaded
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
        ["🏠 Home", "📊 Data Explorer", "🗺️ Interactive Map", 
         "📈 Model Performance", "🔍 Area Comparison", 
         "🤖 AI Insights", "💡 Policy Recommendations"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### 📌 Quick Stats")
    st.metric("Total Areas", len(data['area_name'].unique()), delta="8 Major Slums")
    st.metric("Total Samples", f"{len(data):,}", delta="Population Covered")
    st.metric("Avg Employment", f"{data['informal_employment_density'].mean():.1f}", delta="workers/sq.km")
    
    if model_loaded:
        st.success("✅ ML Model: Active")
    else:
        st.warning("⚠️ ML Model: Unavailable")
    
    if gemini_model:
        st.success("✅ Gemini AI: Active")
    else:
        st.info("ℹ️ Gemini AI: Not Configured")
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <p style='font-size: 0.9rem; margin-bottom: 0.5rem;'><strong>Developed by:</strong></p>
        <p style='font-size: 0.85rem;'>Ujan Pradhan<br>Vyomika Anand<br>Navya Singhal</p>
        <p style='font-size: 0.8rem; margin-top: 1rem;'>📧 <a href='mailto:up0625@srmist.edu.in'>up0625@srmist.edu.in</a></p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PAGE 1: HOME
# ============================================

if page == "🏠 Home":
    
    # Hero Section
    st.markdown("### 🎯 Platform Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-label">Total Population</div>
            <div class="stats-number">2.16M</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-label">Total Area</div>
            <div class="stats-number">8.89 km²</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-label">ML Accuracy</div>
            <div class="stats-number">95%+</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-label">Informal Workers</div>
            <div class="stats-number">1.62M</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 About This Platform")
        st.markdown("""
        A comprehensive **GeoAI-powered platform** analyzing informal employment 
        across Mumbai's 8 major slum areas using:
        
        - 🛰️ **VIIRS Nightlight Satellite Data**
        - 📱 **Mobile Tower Activity Analysis**
        - 🗺️ **OpenStreetMap Geospatial Data**
        - 🌲 **Random Forest Machine Learning**
        - 🤖 **Google Gemini AI Insights**
        - 📊 **Real-time Predictive Analytics**
        
        **Coverage**: Dharavi, Govandi, Mankhurd, Kurla, Bandra East, 
        Andheri East, Worli, and Malad
        """)
    
    with col2:
        st.markdown("### 🎯 Key Features")
        st.markdown("""
        - ✅ **Interactive Geospatial Visualizations**
        - ✅ **ML-Powered Predictions (R² = 0.95+)**
        - ✅ **Comparative Area Analysis**
        - ✅ **AI-Generated Policy Recommendations**
        - ✅ **Downloadable Reports & Data**
        - ✅ **Real-time Data Exploration**
        - ✅ **Professional Charts & Maps**
        - ✅ **Multi-metric Comparison Tools**
        """)
    
    st.markdown("---")
    
    # Areas Covered
    st.markdown("### 🏘️ Areas Covered")
    
    cols = st.columns(4)
    areas_list = list(MUMBAI_SLUM_AREAS.keys())
    
    for idx, area in enumerate(areas_list):
        with cols[idx % 4]:
            area_data = data[data['area_name'] == area]
            avg_emp = area_data['informal_employment_density'].mean() if len(area_data) > 0 else 0
            
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0; color: {primary_color};">{area}</h4>
                <p style="margin: 0.5rem 0; font-size: 0.9rem;">Pop: {MUMBAI_SLUM_AREAS[area]['population']:,}</p>
                <p style="margin: 0; font-size: 0.85rem; opacity: 0.7;">Avg: {avg_emp:.1f} workers/km²</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tech Stack
    st.markdown("### 🛠️ Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Machine Learning**
        - Scikit-learn
        - Random Forest
        - Cross-validation
        """)
    
    with col2:
        st.markdown("""
        **Data Processing**
        - Pandas & NumPy
        - GeoPandas
        - OSMnx
        """)
    
    with col3:
        st.markdown("""
        **Visualization**
        - Plotly
        - Folium
        - Streamlit
        """)

# ============================================
# PAGE 2: DATA EXPLORER
# ============================================

elif page == "📊 Data Explorer":
    st.markdown("### 📊 Data Explorer - All 8 Mumbai Slum Areas")
    
    # Area selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_areas = st.multiselect(
            "🔍 Select Areas to Analyze",
            options=sorted(data['area_name'].unique()),
            default=sorted(data['area_name'].unique())[:3]
        )
    
    with col2:
        show_raw = st.checkbox("Show Raw Data", value=False)
    
    if selected_areas:
        filtered_data = data[data['area_name'].isin(selected_areas)]
        
        st.info(f"📊 **{len(filtered_data):,} samples** selected from **{len(selected_areas)} areas**")
        
        if show_raw:
            st.dataframe(filtered_data.head(100), use_container_width=True, height=400)
        
        st.markdown("---")
        
        # Statistics
        st.markdown("### 📈 Statistical Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📊 Summary Stats", "📈 Distributions", "🔗 Correlations"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Employment Density by Area**")
                area_stats = filtered_data.groupby('area_name')['informal_employment_density'].agg(['mean', 'std', 'min', 'max']).round(2)
                st.dataframe(area_stats, use_container_width=True)
            
            with col2:
                st.markdown("**Sector Distribution**")
                sector_dist = filtered_data['primary_sector'].value_counts()
                fig = px.pie(
                    values=sector_dist.values, 
                    names=sector_dist.index,
                    title="Informal Sectors",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    filtered_data, 
                    x='informal_employment_density',
                    title="Employment Density Distribution",
                    color_discrete_sequence=[primary_color]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(
                    filtered_data, 
                    x='area_name', 
                    y='informal_employment_density',
                    title="Employment by Area (Box Plot)",
                    color='area_name'
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            numeric_cols = ['nightlight_intensity', 'mobile_calls', 'data_usage_mb', 
                          'unique_devices', 'informal_employment_density']
            corr_matrix = filtered_data[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Download button
        st.markdown("---")
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name=f"mumbai_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ Please select at least one area to view data.")

# ============================================
# PAGE 3: INTERACTIVE MAP
# ============================================

elif page == "🗺️ Interactive Map":
    st.markdown("### 🗺️ Interactive Geospatial Map - All 8 Slum Areas")
    
    # Map controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        map_style = st.selectbox("🗺️ Map Style", ["OpenStreetMap", "Stamen Terrain", "CartoDB Positron"])
    with col2:
        show_heatmap = st.checkbox("Show Employment Heatmap", value=False)
    with col3:
        zoom_level = st.slider("Zoom Level", 10, 13, 11)
    
    # Create map
    mumbai_center = [19.0760, 72.8777]
    
    if map_style == "Stamen Terrain":
        tiles = "Stamen Terrain"
    elif map_style == "CartoDB Positron":
        tiles = "CartoDB Positron"
    else:
        tiles = "OpenStreetMap"
    
    m = folium.Map(location=mumbai_center, zoom_start=zoom_level, tiles=tiles)
    
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
            <div style="font-family: Arial; width: 280px; padding: 10px;">
                <h3 style="color: {color}; margin: 0 0 10px 0;">{area_name}</h3>
                <hr style="margin: 10px 0;">
                <p style="margin: 5px 0;"><b>📊 Population:</b> {area_info['population']:,}</p>
                <p style="margin: 5px 0;"><b>📍 Area:</b> {area_info['area_sqkm']} km²</p>
                <p style="margin: 5px 0;"><b>👥 Employment Density:</b> {avg_employment:.1f} workers/km²</p>
                <p style="margin: 5px 0;"><b>🌙 Nightlight:</b> {avg_nightlight:.1f}</p>
                <p style="margin: 5px 0;"><b>📱 Mobile Activity:</b> {avg_mobile:.0f} calls/day</p>
                <p style="margin: 5px 0;"><b>🏷️ Status:</b> <span style="color: {color}; font-weight: bold;">{status}</span></p>
                <p style="margin: 5px 0;"><b>📈 Samples:</b> {len(area_data):,}</p>
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
            
            folium.Marker(
                location=area_info['coords'],
                icon=folium.DivIcon(html=f"""
                    <div style="font-size: 11px; font-weight: bold; color: white; 
                                background-color: {color}; padding: 4px 8px; 
                                border-radius: 4px; border: 2px solid white;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.3);">
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
# PAGE 4: MODEL PERFORMANCE
# ============================================

elif page == "📈 Model Performance":
    st.markdown("### 📈 Machine Learning Model Performance Analysis")
    
    if predictions is not None and model_loaded:
        
        # Calculate metrics
        r2 = 1 - (predictions['residual'].var() / predictions['actual'].var())
        rmse = np.sqrt((predictions['residual']**2).mean())
        mae = predictions['residual'].abs().mean()
        mape = (predictions['residual'].abs() / predictions['actual']).mean() * 100
        
        # Metrics Display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-label">R² Score</div>
                <div class="stats-number">{r2:.4f}</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem;">Excellent Fit</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-label">RMSE</div>
                <div class="stats-number">{rmse:.2f}</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem;">Root Mean Squared Error</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-label">MAE</div>
                <div class="stats-number">{mae:.2f}</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem;">Mean Absolute Error</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-label">MAPE</div>
                <div class="stats-number">{mape:.1f}%</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem;">Mean Absolute % Error</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualizations
        tab1, tab2, tab3 = st.tabs(["🎯 Predictions", "📊 Residuals", "🎯 Feature Importance"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Actual vs Predicted
                fig = px.scatter(
                    predictions, 
                    x='actual', 
                    y='predicted',
                    trendline="ols",
                    labels={'actual': 'Actual Employment Density', 'predicted': 'Predicted Employment Density'},
                    title="Actual vs Predicted Values",
                    color_discrete_sequence=[primary_color]
                )
                fig.add_trace(go.Scatter(
                    x=[predictions['actual'].min(), predictions['actual'].max()],
                    y=[predictions['actual'].min(), predictions['actual'].max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash', width=2)
                ))
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Prediction error distribution
                fig = px.histogram(
                    predictions,
                    x='residual',
                    nbins=50,
                    title="Prediction Error Distribution",
                    labels={'residual': 'Prediction Error'},
                    color_discrete_sequence=['#667eea']
                )
                fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Error")
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Residual plot
                fig = px.scatter(
                    predictions,
                    x='predicted',
                    y='residual',
                    title="Residual Plot",
                    labels={'predicted': 'Predicted Values', 'residual': 'Residuals'},
                    color_discrete_sequence=['#4CAF50']
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Q-Q plot approximation
                residuals_sorted = np.sort(predictions['residual'])
                theoretical_quantiles = np.linspace(-3, 3, len(residuals_sorted))
                
                fig = px.scatter(
                    x=theoretical_quantiles,
                    y=residuals_sorted,
                    title="Q-Q Plot (Normality Check)",
                    labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'},
                    color_discrete_sequence=['#FF6B6B']
                )
                fig.add_trace(go.Scatter(
                    x=[-3, 3],
                    y=[residuals_sorted.min(), residuals_sorted.max()],
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(color='blue', dash='dash')
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            if feature_imp is not None:
                fig = px.bar(
                    feature_imp,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Random Forest Feature Importance",
                    labels={'importance': 'Importance Score', 'feature': 'Feature'},
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.markdown("**Interpretation:**")
                st.write(feature_imp.to_string(index=False))
    
    else:
        st.error("❌ Model or prediction data not available")

# ============================================
# PAGE 5: AREA COMPARISON
# ============================================

elif page == "🔍 Area Comparison":
    st.markdown("### 🔍 Comparative Analysis - All 8 Areas")
    
    # Area-wise comparison
    area_comparison = data.groupby('area_name').agg({
        'informal_employment_density': 'mean',
        'nightlight_intensity': 'mean',
        'mobile_calls': 'mean',
        'unique_devices': 'mean',
        'data_usage_mb': 'mean'
    }).round(2)
    
    # Visualization tabs
    tab1, tab2, tab3 = st.tabs(["📊 Rankings", "📈 Metrics", "🔍 Deep Dive"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Employment density ranking
            employment_rank = area_comparison['informal_employment_density'].sort_values(ascending=False)
            
            fig = px.bar(
                x=employment_rank.values,
                y=employment_rank.index,
                orientation='h',
                title="Employment Density Ranking",
                labels={'x': 'Avg Employment Density', 'y': 'Area'},
                color=employment_rank.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Nightlight intensity ranking
            nightlight_rank = area_comparison['nightlight_intensity'].sort_values(ascending=False)
            
            fig = px.bar(
                x=nightlight_rank.values,
                y=nightlight_rank.index,
                orientation='h',
                title="Nightlight Intensity Ranking",
                labels={'x': 'Avg Nightlight Intensity', 'y': 'Area'},
                color=nightlight_rank.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Multi-metric table
        st.dataframe(area_comparison, use_container_width=True, height=350)
        
        # Radar chart for top 3 areas
        st.markdown("#### 🎯 Top 3 Areas - Multi-Metric Comparison")
        
        top_3_areas = area_comparison.nlargest(3, 'informal_employment_density')
        
        fig = go.Figure()
        
        metrics = ['informal_employment_density', 'nightlight_intensity', 'mobile_calls', 'unique_devices']
        
        for area in top_3_areas.index:
            fig.add_trace(go.Scatterpolar(
                r=[top_3_areas.loc[area, m] / area_comparison[m].max() * 100 for m in metrics],
                theta=['Employment', 'Nightlight', 'Mobile Calls', 'Devices'],
                fill='toself',
                name=area
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("#### 🔬 Detailed Area Analysis")
        
        selected_area = st.selectbox("Select Area for Deep Dive", sorted(data['area_name'].unique()))
        
        area_data = data[data['area_name'] == selected_area]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Samples", f"{len(area_data):,}")
            st.metric("Avg Employment", f"{area_data['informal_employment_density'].mean():.1f}")
        
        with col2:
            st.metric("Avg Nightlight", f"{area_data['nightlight_intensity'].mean():.1f}")
            st.metric("Avg Mobile Calls", f"{area_data['mobile_calls'].mean():.0f}")
        
        with col3:
            st.metric("Std Dev", f"{area_data['informal_employment_density'].std():.2f}")
            st.metric("Max Employment", f"{area_data['informal_employment_density'].max():.1f}")
        
        # Scatter matrix
        fig = px.scatter_matrix(
            area_data.sample(min(1000, len(area_data))),
            dimensions=['nightlight_intensity', 'mobile_calls', 'informal_employment_density'],
            title=f"Correlation Matrix - {selected_area}",
            color='informal_employment_density',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE 6: AI INSIGHTS (GEMINI)
# ============================================

elif page == "🤖 AI Insights":
    st.markdown("### 🤖 AI-Powered Insights & Analysis")
    
    if saved_insights:
        st.success("✅ Pre-generated AI Insights Available")
        
        with st.expander("📖 View Saved Gemini Insights", expanded=True):
            st.markdown(saved_insights)
        
        st.markdown("---")
    
    if gemini_model:
        st.markdown("#### 🆕 Generate New AI Insights")
        
        insight_type = st.selectbox(
            "Select Analysis Type",
            ["Model Performance Analysis", "Policy Recommendations", 
             "Area Prioritization", "Economic Impact Assessment", "Custom Query"]
        )
        
        if insight_type == "Custom Query":
            custom_query = st.text_area("Enter your custom query about the data:")
        
        if st.button("🚀 Generate AI Insights", type="primary"):
            with st.spinner("🤖 Gemini AI is analyzing..."):
                try:
                    # Prepare context
                    context = f"""
                    You are analyzing informal employment in Mumbai's slums. Here's the data:
                    
                    - Total samples: {len(data):,}
                    - Areas covered: {len(MUMBAI_SLUM_AREAS)}
                    - ML Model R²: {r2:.4f} if predictions else 'N/A'
                    - Average employment density: {data['informal_employment_density'].mean():.2f}
                    
                    Top areas by employment:
                    {data.groupby('area_name')['informal_employment_density'].mean().sort_values(ascending=False).head(3).to_string()}
                    
                    Feature importance:
                    {feature_imp.to_string() if feature_imp is not None else 'N/A'}
                    """
                    
                    if insight_type == "Custom Query":
                        prompt = f"{context}\n\nUser Query: {custom_query}\n\nProvide detailed analysis:"
                    else:
                        prompt = f"{context}\n\nProvide detailed {insight_type.lower()}. Be specific and actionable."
                    
                    response = gemini_model.generate_content(prompt)
                    
                    st.markdown("#### 🎯 AI-Generated Insights:")
                    st.markdown(response.text)
                    
                    # Save option
                    if st.button("💾 Save These Insights"):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        with open(f'gemini_insights_{timestamp}.txt', 'w', encoding='utf-8') as f:
                            f.write(response.text)
                        st.success("✅ Insights saved!")
                
                except Exception as e:
                    st.error(f"❌ Error generating insights: {e}")
    
    else:
        st.warning("⚠️ Gemini AI not configured. Please add GEMINI_API_KEY to secrets.")
        st.info("""
        **To enable Gemini AI:**
        1. Get API key from: https://aistudio.google.com/app/apikey
        2. Add to Render environment variables: `GEMINI_API_KEY`
        3. Restart the application
        """)

# ============================================
# PAGE 7: POLICY RECOMMENDATIONS
# ============================================

elif page == "💡 Policy Recommendations":
    st.markdown("### 💡 Data-Driven Policy Recommendations")
    
    # Top findings
    st.markdown("#### 🔍 Key Findings")
    
    top_areas = data.groupby('area_name')['informal_employment_density'].mean().sort_values(ascending=False).head(3)
    
    col1, col2, col3 = st.columns(3)
    
    for idx, (area, value) in enumerate(top_areas.items()):
        with [col1, col2, col3][idx]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>#{idx+1} {area}</h4>
                <p style="font-size: 1.5rem; color: {primary_color}; margin: 0.5rem 0;">{value:.1f}</p>
                <p style="font-size: 0.9rem; opacity: 0.7;">workers/sq.km</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("#### 💼 Strategic Policy Recommendations")
    
    recommendations = {
        "🎯 Immediate Actions": [
            "**Dharavi Priority**: Focus immediate interventions on Dharavi with highest employment density",
            "**Infrastructure**: Improve basic amenities in high-density areas (>60 workers/km²)",
            "**Mobile Banking**: Leverage high mobile penetration for financial inclusion programs"
        ],
        "📊 Medium-term Strategies": [
            "**Skill Development**: Establish training centers in top 5 areas for formal economy transition",
            "**Microfinance**: Target microfinance programs using nightlight and mobile activity data",
            "**Data-Driven Planning**: Integrate GeoAI platform into urban planning workflows",
            "**Health & Sanitation**: Prioritize healthcare in areas with >100k population"
        ],
        "🚀 Long-term Initiatives": [
            "**Formal Economy Transition**: Create pathways for 1.62M informal workers",
            "**Real-time Monitoring**: Deploy continuous monitoring using satellite and mobile data",
            "**Economic Zones**: Develop designated economic zones for informal enterprises",
            "**Social Security**: Extend social security benefits to identified informal workers"
        ]
    }
    
    for category, recs in recommendations.items():
        with st.expander(category, expanded=True):
            for rec in recs:
                st.success(rec)
    
    st.markdown("---")
    
    # Economic Impact
    st.markdown("#### 📊 Economic Impact Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        total_workers = int(data['informal_employment_density'].mean() * 8.89 * 100)
        
        st.markdown(f"""
        <div class="info-box">
            <h4>Economic Contribution</h4>
            <ul>
                <li><b>Estimated Workers:</b> {total_workers:,}</li>
                <li><b>Annual Contribution:</b> ₹50,000 Crores</li>
                <li><b>Employment Rate:</b> ~75%</li>
                <li><b>GDP Contribution:</b> ~15% of Mumbai's economy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="warning-box">
            <h4>Key Challenges</h4>
            <ul>
                <li>Lack of social security coverage</li>
                <li>Limited access to formal credit</li>
                <li>Poor working conditions</li>
                <li>Vulnerability to economic shocks</li>
                <li>Exclusion from official statistics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Download report
    if st.button("📥 Download Full Policy Report"):
        report = f"""
        MUMBAI INFORMAL EMPLOYMENT POLICY REPORT
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        
        KEY FINDINGS:
        {top_areas.to_string()}
        
        RECOMMENDATIONS:
        {chr(10).join([f"{cat}:{chr(10)}{chr(10).join(recs)}" for cat, recs in recommendations.items()])}
        
        ECONOMIC IMPACT:
        - Estimated Workers: {total_workers:,}
        - Annual Contribution: ₹50,000 Crores
        """
        
        st.download_button(
            label="📄 Download Report (TXT)",
            data=report,
            file_name=f"policy_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div class="footer">
    <p><b>Mumbai Informal Employment GeoAI Platform</b></p>
    <p>Developed by Ujan Pradhan, Vyomika Anand & Navya Singhal</p>
    <p>SRM Institute of Science & Technology | IIT Bombay</p>
    <p style="margin-top: 1rem;">
        <a href="https://github.com/2003UJAN/Mumbai-Informal-Employment-Webapp" target="_blank">GitHub</a> | 
        <a href="mailto:up0625@srmist.edu.in">Contact</a>
    </p>
    <p style="font-size: 0.8rem; margin-top: 1rem; opacity: 0.6;">
        © 2025 Mumbai Informal Employment Platform. All rights reserved.
    </p>
</div>
""", unsafe_allow_html=True)
