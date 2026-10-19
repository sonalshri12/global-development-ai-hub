import numpy as np
# Compatibility patches for newer NumPy versions breaking older libraries
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
if not hasattr(np, 'round_'):
    np.round_ = np.round
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="World Development Cluster Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 World Development Measurement — Cluster Analysis Dashboard")
st.markdown("""
This interactive deployment dashboard groups global economies into socio-economic development clusters 
using **$K$-Means Clustering** based on health, economic, and social indicators.
""")

# ==========================================
# 2. DATA LOADING & PREPROCESSING PIPELINE
# ==========================================
@st.cache_data
def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path)
    
    # Strip hidden whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Clean monetary and percentage strings across all text/object columns
    money_cols = ['GDP', 'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound', 'Business Tax Rate']
    for col in money_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[\$,% ]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop redundant columns
    df = df.drop(columns=['Ease of Business', 'Number of Records'], errors='ignore')

    # Aggregate by Country
    country_df = df.groupby('Country').mean(numeric_only=True)

    # Feature Engineering (Per Capita)
    pop = country_df['Population Total']
    country_df['GDP_per_Capita'] = country_df['GDP'] / pop
    country_df['CO2_per_Capita'] = country_df['CO2 Emissions'] / pop
    country_df['Energy_per_Capita'] = country_df['Energy Usage'] / pop
    country_df['Tourism_Inbound_per_Capita'] = country_df['Tourism Inbound'] / pop
    country_df['Tourism_Outbound_per_Capita'] = country_df['Tourism Outbound'] / pop

    # Select features for clustering
    raw_totals = ['GDP', 'CO2 Emissions', 'Energy Usage', 'Tourism Inbound', 'Tourism Outbound', 'Population Total']
    cluster_features = [col for col in country_df.columns if col not in raw_totals]

    X = country_df[cluster_features].copy()
    X_imputed = X.fillna(X.median())

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return country_df, cluster_features, X_imputed, X_scaled

# Load dataset safely
DATA_FILE = 'P706_ World_development_mesurement.xlsx'

try:
    country_df, cluster_features, X_imputed, X_scaled = load_and_preprocess_data(DATA_FILE)
except Exception as e:
    st.error(f"Error loading dataset: {e}. Make sure '{DATA_FILE}' is in the same folder as app.py.")
    st.stop()

# ==========================================
# 3. MODEL TRAINING & DEPLOYMENT SAVING
# ==========================================
n_clusters = st.sidebar.slider("Select Number of Clusters (K):", min_value=2, max_value=6, value=3)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

country_df['Cluster'] = [f"Cluster {l}" for l in labels]

# Save model & scaler artifacts for deployment demonstration
scaler_deployment = StandardScaler()
scaler_deployment.fit(X_imputed)
joblib.dump(kmeans, 'kmeans_deployed_model.pkl')
joblib.dump(scaler_deployment, 'scaler_deployed.pkl')

# ==========================================
# 4. DASHBOARD INTERFACE TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Cluster Overview", "📈 Interactive Visualizations", "📋 Country Data Table", "🚀 Model Deployment / Predictor"])

with tab1:
    st.subheader("Key Summary Metrics Across Clusters")
    summary_cols = ['GDP_per_Capita', 'Health Exp/Capita', 'Life Expectancy Female', 
                    'Infant Mortality Rate', 'Internet Usage', 'CO2_per_Capita']
    
    cluster_summary = country_df.groupby('Cluster')[summary_cols].mean()
    st.dataframe(cluster_summary.style.format("{:,.2f}").highlight_max(axis=0, color="#d1e7dd"), use_container_width=True)

with tab2:
    st.subheader("2D Cluster Scatter Plot")
    col_x, col_y = st.columns(2)
    with col_x:
        x_axis = st.selectbox("X Axis Feature:", options=cluster_features, index=cluster_features.index('GDP_per_Capita'))
    with col_y:
        y_axis = st.selectbox("Y Axis Feature:", options=cluster_features, index=cluster_features.index('Life Expectancy Female'))

    filtered_df = country_df[country_df['Cluster'].isin(country_df['Cluster'].unique())]
    fig_scatter = px.scatter(
        filtered_df.reset_index(),
        x=x_axis,
        y=y_axis,
        color='Cluster',
        hover_name='Country',
        title=f"{y_axis} vs {x_axis}",
        template="plotly_white",
        height=550
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.subheader("Explore Country Assignments")
    st.dataframe(country_df[['Cluster', 'GDP_per_Capita', 'Health Exp/Capita', 'Life Expectancy Female', 'Internet Usage']], use_container_width=True)
    
    csv_data = country_df.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Clustered Dataset (CSV)",
        data=csv_data,
        file_name='world_development_clusters.csv',
        mime='text/csv'
    )

with tab4:
    st.subheader("🚀 Real-Time Model Deployment Inference")
    st.markdown("Test the deployed model by passing feature values or selecting an existing country profile to see its predicted cluster assignment.")
    
    selected_country = st.selectbox("Select a country to test prediction:", options=country_df.index.sort_values())
    
    if st.button("Predict Cluster for Selected Country"):
        loaded_model = joblib.load('kmeans_deployed_model.pkl')
        loaded_scaler = joblib.load('scaler_deployed.pkl')
        
        country_data_raw = X_imputed.loc[[selected_country]]
        scaled_input = loaded_scaler.transform(country_data_raw)
        pred_cluster = loaded_model.predict(scaled_input)[0]
        
        st.success(f"✅ **{selected_country}** is classified under **Cluster {pred_cluster}** using the deployed artifact files (`.pkl`).")
