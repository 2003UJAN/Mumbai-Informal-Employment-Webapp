import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import pickle
from datetime import datetime

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
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

# -----------------------------------------------------
# LOAD DATASET (OPTIMIZED FOR LARGE FILES)
# -----------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv('data/mumbai_all_8_areas_data.csv', low_memory=False)
    stats = pd.read_csv('data/mumbai_all_areas_statistics.csv', low_memory=False)
    predictions = pd.read_csv('data/mumbai_all_areas_predictions.csv', low_memory=False)
    feature_imp = pd.read_csv('data/mumbai_feature_importance.csv', low_memory=False)
    return data, stats, predictions, feature_imp

# Load model
@st.cache_resource
def load_model():
    with open('models/mumbai_rf_model_all_areas.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# -----------------------------------------------------
# LOAD EVERYTHING WITH SPINNER
# -----------------------------------------------------
with st.spinner("📥 Loading dataset & model..."):
    try:
        data, stats, predictions, feature_imp = load_data()
        model = load_model()
    except:
        st.error("❌ Error loading files. Ensure data/ and models/ folders exist.")
        st.stop()

# Title
st.markdown('<h1 class="main-header">🏙️ Mumbai Informal Employment Analysis</h1>', unsafe_allow_html=True)
st.markdown("**Geospatial Econometrics of Informal Employment in Mumbai's 8 Major Slum Areas**")

# Sidebar
st.sidebar.title("🗺️ Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📊 Data Explorer", "🗺️ Interactive Map", "📈 Model Performance", "🔍 Area Comparison", "💡 Insights & Recommendations"]
)

st.sidebar.markdown("---")
st.sidebar.header("📌 Quick Stats")
st.sidebar.metric("Total Areas", len(data['area_name'].unique()))
st.sidebar.metric("Total Samples", f"{len(data):,}")
st.sidebar.metric("Avg Employment", f"{data['informal_employment_density'].mean():.1f}")

# Mumbai area config
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

# -----------------------------------------------------
# PAGE 1: HOME
# -----------------------------------------------------
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

    st.subheader("📊 Key Metrics - All 8 Areas")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Population", "2,160,000")
    col2.metric("Total Area", "8.89 sq.km")
    col3.metric("Avg Density", "242,968/sq.km")
    col4.metric("Est. Informal Workers", "1,620,000")

    st.divider()

    st.subheader("🏘️ Areas Covered")
    cols = st.columns(4)
    for i, area in enumerate(MUMBAI_SLUM_AREAS.keys()):
        cols[i % 4].info(f"**{area}**\n\nPop: {MUMBAI_SLUM_AREAS[area]['population']:,}")

# -----------------------------------------------------
# PAGE 2: DATA EXPLORER (PAGINATION ✅)
# -----------------------------------------------------
elif page == "📊 Data Explorer":
    st.header("Data Explorer - All 8 Mumbai Slum Areas")

    selected_areas = st.multiselect(
        "Select Areas to View",
        options=data['area_name'].unique(),
        default=data['area_name'].unique()[:3]
    )

    if selected_areas:
        filtered_data = data[data['area_name'].isin(selected_areas)]

        st.subheader(f"Data Preview ({len(filtered_data):,} samples)")

        # Pagination (100 rows per page)
        rows_per_page = 100
        total_pages = (len(filtered_data) // rows_per_page) + 1
        page_no = st.number_input("Page", 1, total_pages)

        start = (page_no - 1) * rows_per_page
        end = start + rows_per_page

        st.dataframe(filtered_data.iloc[start:end], use_container_width=True)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Employment Density by Area**")
            area_stats = filtered_data.groupby('area_name')['informal_employment_density'].agg(['mean', 'std', 'min', 'max']).round(2)
            st.dataframe(area_stats, use_container_width=True)

        with col2:
            st.write("**Sector Distribution**")
            fig = px.pie(values=filtered_data['primary_sector'].value_counts().values,
                         names=filtered_data['primary_sector'].value_counts().index)
            st.plotly_chart(fig, use_container_width=True)

        csv = filtered_data.to_csv(index=False)
        st.download_button(
            "📥 Download Filtered Data",
            csv,
            file_name=f"mumbai_data_{datetime.now().strftime('%Y%m%d')}.csv"
        )
    else:
        st.warning("Please select at least one area.")

# -----------------------------------------------------
# PAGE 3: INTERACTIVE MAP
# -----------------------------------------------------
elif page == "🗺️ Interactive Map":
    st.header("Interactive Map - All 8 Slum Areas")

    with st.spinner("Building map..."):
        mumbai_center = [19.0760, 72.8777]
        m = folium.Map(location=mumbai_center, zoom_start=11)

        for area, details in MUMBAI_SLUM_AREAS.items():
            subset = data[data['area_name'] == area]
            if len(subset) > 0:
                avg_emp = subset['informal_employment_density'].mean()
                color = "red" if avg_emp > 60 else "orange" if avg_emp > 40 else "green"

                popup_html = f"""
                <b>{area}</b><br>
                Population: {details['population']:,}<br>
                Avg Employment: {avg_emp:.1f}
                """
                folium.CircleMarker(details['coords'], radius=15, color=color, fill=True, fillOpacity=0.6, popup=popup_html).add_to(m)

        st_folium(m, height=600, width=1200)

    st.markdown("""
    **Legend:**
    - 🔴 High (>60 workers/sq.km)
    - 🟠 Medium (40-60 workers/sq.km)
    - 🟢 Low (<40 workers/sq.km)
    """)

# -----------------------------------------------------
# PAGE 4: MODEL PERFORMANCE
# -----------------------------------------------------
elif page == "📈 Model Performance":
    st.header("Model Performance Analysis")

    if predictions is not None:
        st.subheader("📊 Performance Metrics")

        residuals = predictions['residual']
        actual = predictions['actual']

        r2 = 1 - (residuals.var() / actual.var())
        rmse = np.sqrt((residuals ** 2).mean())
        mae = residuals.abs().mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{r2:.4f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("MAE", f"{mae:.2f}")

        st.divider()
        st.subheader("🎯 Actual vs Predicted")

        fig = px.scatter(predictions, x="actual", y="predicted", title="Model Predictions")
        fig.add_trace(go.Scatter(x=[actual.min(), actual.max()], y=[actual.min(), actual.max()], mode='lines', name='Perfect Prediction'))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("🎯 Feature Importance")

        fig = px.bar(feature_imp, x="importance", y="feature", orientation='h')
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# PAGE 5: AREA COMPARISON
# -----------------------------------------------------
elif page == "🔍 Area Comparison":
    st.header("Comparative Analysis - All Areas")

    st.subheader("📊 Employment Density Comparison")
    area_comparison = data.groupby('area_name')['informal_employment_density'].mean().sort_values(ascending=False)

    fig = px.bar(x=area_comparison.values, y=area_comparison.index, orientation='h', title="Average Informal Employment by Area")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("📈 Multi-Metric Comparison")

    comparison_metrics = data.groupby('area_name').agg({
        'informal_employment_density': 'mean',
        'nightlight_intensity': 'mean',
        'mobile_calls': 'mean',
        'unique_devices': 'mean'
    }).round(2)

    st.dataframe(comparison_metrics, use_container_width=True)

    st.divider()
    st.subheader("🔍 Correlation Analysis")

    selected_area = st.selectbox("Select Area", data['area_name'].unique())
    area_data = data[data['area_name'] == selected_area]

    fig = px.scatter_matrix(area_data, dimensions=['nightlight_intensity', 'mobile_calls', 'informal_employment_density'])
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# PAGE 6: INSIGHTS & RECOMMENDATIONS
# -----------------------------------------------------
elif page == "💡 Insights & Recommendations":
    st.header("Key Insights & Policy Recommendations")

    st.subheader("🔍 Key Findings")

    top_areas = data.groupby('area_name')['informal_employment_density'].mean().sort_values(ascending=False).head(3)
    for i, (area, value) in enumerate(top_areas.items(), 1):
        st.write(f"{i}. **{area}**: {value:.1f} workers/sq.km")

    st.divider()

    st.subheader("💼 Policy Recommendations")
    recommendations = [
        "**Targeted Interventions**: Focus on high-density areas (Dharavi, Bandra East)",
        "**Infrastructure Development**: Improve basic amenities",
        "**Skill Development**: Training centers for formalization",
        "**Financial Inclusion**: Expand microfinance + banking access",
        "**Data-Driven Planning**: Use real-time geospatial decision-making"
    ]

    for rec in recommendations:
        st.success(rec)

    st.divider()

    st.subheader("📊 Economic Impact")

    total_workers = int(data['informal_employment_density'].mean() * 8.89 * 100)

    col1, col2 = st.columns(2)
    col1.info(f"""
    **Estimated Informal Workers**: {total_workers:,}
    
    **Estimated Annual Contribution**: ₹50,000 Crores
    **Employment Rate**: ~75%
    """)
    col2.warning("""
    **Challenges**
    - Lack of social security
    - Limited access to credit
    - Poor working conditions
    - Vulnerability to economic shocks
    """)

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.info("""
**Developed by:**
Ujan Pradhan , Vyomika Anand & Navya Singhal

📧 up0625@srmist.edu.in
""")
