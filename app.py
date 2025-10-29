import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import pickle
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Mumbai Informal Employment Analysis",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('data/mumbai_all_8_areas_data.csv')
        stats = pd.read_csv('data/mumbai_all_areas_statistics.csv')
        predictions = pd.read_csv('data/mumbai_all_areas_predictions.csv')
        feature_imp = pd.read_csv('data/mumbai_feature_importance.csv')
        return data, stats, predictions, feature_imp
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("📁 Please ensure CSV files are in the 'data/' folder")
        return None, None, None, None

# Load model
@st.cache_resource
def load_model():
    try:
        with open('models/mumbai_rf_model_all_areas.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.warning(f"Model not loaded: {e}")
        st.info("📁 Please ensure model file is in the 'models/' folder")
        return None

# Title
st.markdown('<h1 class="main-header">🏙️ Mumbai Informal Employment Analysis</h1>', unsafe_allow_html=True)
st.markdown("**Geospatial Econometrics of Informal Employment in Mumbai's 8 Major Slum Areas**")

# Load data
data, stats, predictions, feature_imp = load_data()
model = load_model()

if data is None:
    st.error("Failed to load data files. Please ensure all CSV files are in the same directory.")
    st.stop()

# Sidebar
st.sidebar.title("🗺️ Navigation")
page = st.sidebar.radio("Select Page", [
    "🏠 Home",
    "📊 Data Explorer",
    "🗺️ Interactive Map",
    "📈 Model Performance",
    "🔍 Area Comparison",
    "💡 Insights & Recommendations"
])

st.sidebar.markdown("---")
st.sidebar.header("📌 Quick Stats")
st.sidebar.metric("Total Areas", len(data['area_name'].unique()))
st.sidebar.metric("Total Samples", f"{len(data):,}")
st.sidebar.metric("Avg Employment", f"{data['informal_employment_density'].mean():.1f}")

# Mumbai areas configuration
MUMBAI_SLUM_AREAS = {
    'Dharavi': {'coords': (19.0444, 72.8560), 'population': 1000000},
    'Govandi': {'coords': (19.0545, 72.9144), 'population': 250000},
    'Mankhurd': {'coords': (19.0437, 72.9264), 'population': 200000},
    'Kurla': {'coords': (19.0728, 72.8826), 'population': 150000},
    'Bandra East': {'coords': (19.0596, 72.8428), 'population': 120000},
    'Andheri East': {'coords': (19.1136, 72.8697), 'population': 180000},
    'Worli': {'coords': (19.0144, 72.8169), 'population': 100000},
    'Malad': {'coords': (19.1864, 72.8479), 'population': 160000}
}

# ============================================
# PAGE 1: HOME
# ============================================

if page == "🏠 Home":
    st.header("Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 About This Project")
        st.write("""
        This application presents a comprehensive geospatial econometric analysis 
        of informal employment across **8 major slum areas in Mumbai**:
        
        - 🌙 **Nightlight Satellite Data**
        - 📱 **Mobile Tower Activity**
        - 🏘️ **OpenStreetMap Land Use**
        - 🌲 **Random Forest ML Model**
        - 📊 **2.16 Million Population Covered**
        """)
        
    with col2:
        st.subheader("🎯 Key Features")
        st.write("""
        - Interactive geospatial visualizations
        - Machine learning predictions
        - Comparative area analysis
        - Policy recommendations
        - Downloadable reports
        """)
    
    st.divider()
    
    # Key Metrics
    st.subheader("📊 Key Metrics - All 8 Areas")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Population", "2,160,000")
    with col2:
        st.metric("Total Area", "8.89 sq.km")
    with col3:
        st.metric("Avg Density", "242,968/sq.km")
    with col4:
        st.metric("Est. Informal Workers", "1,620,000")
    
    st.divider()
    
    # Areas covered
    st.subheader("🏘️ Areas Covered")
    
    cols = st.columns(4)
    areas_list = list(MUMBAI_SLUM_AREAS.keys())
    
    for idx, area in enumerate(areas_list):
        with cols[idx % 4]:
            st.info(f"**{area}**\n\nPop: {MUMBAI_SLUM_AREAS[area]['population']:,}")

# ============================================
# PAGE 2: DATA EXPLORER
# ============================================

elif page == "📊 Data Explorer":
    st.header("Data Explorer - All 8 Mumbai Slum Areas")
    
    # Area selector
    selected_areas = st.multiselect(
        "Select Areas to View",
        options=data['area_name'].unique(),
        default=data['area_name'].unique()[:3]
    )
    
    if selected_areas:
        filtered_data = data[data['area_name'].isin(selected_areas)]
        
        st.subheader(f"Data Preview ({len(filtered_data):,} samples)")
        st.dataframe(filtered_data.head(100), use_container_width=True)
        
        st.divider()
        
        # Statistics
        st.subheader("📈 Statistical Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Employment Density by Area**")
            area_stats = filtered_data.groupby('area_name')['informal_employment_density'].agg(['mean', 'std', 'min', 'max']).round(2)
            st.dataframe(area_stats, use_container_width=True)
        
        with col2:
            st.write("**Sector Distribution**")
            sector_dist = filtered_data['primary_sector'].value_counts()
            fig = px.pie(values=sector_dist.values, names=sector_dist.index, title="Sectors")
            st.plotly_chart(fig, use_container_width=True)
        
        # Download button
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name=f"mumbai_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("Please select at least one area.")

# ============================================
# PAGE 3: INTERACTIVE MAP
# ============================================

elif page == "🗺️ Interactive Map":
    st.header("Interactive Map - All 8 Slum Areas")
    
    # Create map
    mumbai_center = [19.0760, 72.8777]
    m = folium.Map(location=mumbai_center, zoom_start=11)
    
    for area_name, area_info in MUMBAI_SLUM_AREAS.items():
        area_data = data[data['area_name'] == area_name]
        
        if len(area_data) > 0:
            avg_employment = area_data['informal_employment_density'].mean()
            
            if avg_employment > 60:
                color = 'red'
            elif avg_employment > 40:
                color = 'orange'
            else:
                color = 'green'
            
            popup_html = f"""
            <div style="font-family: Arial; width: 250px;">
                <h4 style="color: {color};">{area_name}</h4>
                <p><b>Population:</b> {area_info['population']:,}</p>
                <p><b>Avg Employment:</b> {avg_employment:.1f} workers/sq.km</p>
                <p><b>Samples:</b> {len(area_data):,}</p>
            </div>
            """
            
            folium.CircleMarker(
                location=area_info['coords'],
                radius=15,
                popup=folium.Popup(popup_html, max_width=300),
                color=color,
                fill=True,
                fillOpacity=0.6
            ).add_to(m)
            
            folium.Marker(
                location=area_info['coords'],
                icon=folium.DivIcon(html=f'<div style="font-size: 10px; font-weight: bold;">{area_name}</div>')
            ).add_to(m)
    
    st_folium(m, width=1200, height=600)
    
    # Legend
    st.markdown("""
    **Legend:**
    - 🔴 Red: High (>60 workers/sq.km)
    - 🟠 Orange: Medium (40-60 workers/sq.km)
    - 🟢 Green: Low (<40 workers/sq.km)
    """)

# ============================================
# PAGE 4: MODEL PERFORMANCE
# ============================================

elif page == "📈 Model Performance":
    st.header("Model Performance Analysis")
    
    if predictions is not None:
        # Metrics
        st.subheader("📊 Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        r2 = 1 - (predictions['residual'].var() / predictions['actual'].var())
        rmse = np.sqrt((predictions['residual']**2).mean())
        mae = predictions['residual'].abs().mean()
        
        with col1:
            st.metric("R² Score", f"{r2:.4f}")
        with col2:
            st.metric("RMSE", f"{rmse:.2f}")
        with col3:
            st.metric("MAE", f"{mae:.2f}")
        
        st.divider()
        
        # Actual vs Predicted
        st.subheader("🎯 Actual vs Predicted")
        
        fig = px.scatter(
            predictions, 
            x='actual', 
            y='predicted',
            labels={'actual': 'Actual Employment Density', 'predicted': 'Predicted Employment Density'},
            title="Model Predictions"
        )
        fig.add_trace(go.Scatter(
            x=[predictions['actual'].min(), predictions['actual'].max()],
            y=[predictions['actual'].min(), predictions['actual'].max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Feature Importance
        if feature_imp is not None:
            st.subheader("🎯 Feature Importance")
            
            fig = px.bar(
                feature_imp,
                x='importance',
                y='feature',
                orientation='h',
                title="Random Forest Feature Importance"
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE 5: AREA COMPARISON
# ============================================

elif page == "🔍 Area Comparison":
    st.header("Comparative Analysis - All Areas")
    
    # Employment density comparison
    st.subheader("📊 Employment Density Comparison")
    
    area_comparison = data.groupby('area_name')['informal_employment_density'].mean().sort_values(ascending=False)
    
    fig = px.bar(
        x=area_comparison.values,
        y=area_comparison.index,
        orientation='h',
        labels={'x': 'Avg Informal Employment Density', 'y': 'Area'},
        title="Average Informal Employment by Area"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Multi-metric comparison
    st.subheader("📈 Multi-Metric Comparison")
    
    comparison_metrics = data.groupby('area_name').agg({
        'informal_employment_density': 'mean',
        'nightlight_intensity': 'mean',
        'mobile_calls': 'mean',
        'unique_devices': 'mean'
    }).round(2)
    
    st.dataframe(comparison_metrics, use_container_width=True)
    
    st.divider()
    
    # Scatter matrix
    st.subheader("🔍 Correlation Analysis")
    
    selected_area = st.selectbox("Select Area", data['area_name'].unique())
    area_data = data[data['area_name'] == selected_area]
    
    fig = px.scatter_matrix(
        area_data,
        dimensions=['nightlight_intensity', 'mobile_calls', 'informal_employment_density'],
        title=f"Correlation Matrix - {selected_area}"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE 6: INSIGHTS
# ============================================

elif page == "💡 Insights & Recommendations":
    st.header("Key Insights & Policy Recommendations")
    
    st.subheader("🔍 Key Findings")
    
    # Top 3 areas
    top_areas = data.groupby('area_name')['informal_employment_density'].mean().sort_values(ascending=False).head(3)
    
    st.write("**Top 3 Areas by Informal Employment Density:**")
    for idx, (area, value) in enumerate(top_areas.items(), 1):
        st.write(f"{idx}. **{area}**: {value:.1f} workers/sq.km")
    
    st.divider()
    
    st.subheader("💼 Policy Recommendations")
    
    recommendations = [
        "**Targeted Interventions**: Focus on high-density areas (Dharavi, Bandra East) for immediate policy interventions",
        "**Infrastructure Development**: Improve basic amenities in areas with high informal employment",
        "**Skill Development**: Establish training centers for informal workers to transition to formal economy",
        "**Financial Inclusion**: Expand microfinance and banking services in underserved areas",
        "**Data-Driven Planning**: Use real-time geospatial data for urban planning decisions"
    ]
    
    for rec in recommendations:
        st.success(rec)
    
    st.divider()
    
    st.subheader("📊 Economic Impact")
    
    total_workers = int(data['informal_employment_density'].mean() * 8.89 * 100)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Estimated Informal Workers**: {total_workers:,}
        
        **Estimated Annual Contribution**: ₹50,000 Crores
        
        **Employment Rate**: ~75%
        """)
    
    with col2:
        st.warning("""
        **Challenges**:
        - Lack of social security
        - Limited access to credit
        - Poor working conditions
        - Vulnerability to economic shocks
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Developed by:**
Ujan Pradhan , Vyomika Anand & Navya Singhal

📧 up0625@srmist.edu.in
""")
