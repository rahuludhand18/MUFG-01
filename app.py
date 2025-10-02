import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Manufacturing Output Predictor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .dataframe {
        color: black !important;
    }
    .dataframe th {
        color: black !important;
        background-color: #f0f0f0 !important;
    }
    .dataframe td {
        color: black !important;
    }
    .prediction-result {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1f77b4;
    }
    .insight-box {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv('manufacturing_dataset_1000_samples.csv')
    return df

@st.cache_resource
def load_model():
    try:
        with open('manufacturing_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("Model file not found. Please run the analysis first.")
        return None

# Load resources
df = load_data()
model_data = load_model()

# Sidebar navigation
st.sidebar.markdown('<div class="sidebar-header">🏭 Manufacturing Output Predictor</div>', unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📊 Data Analysis", "🤖 Model Performance", "🔮 Predictions", "💡 Insights"],
    index=0
)

# Main header
st.markdown('<div class="main-header">Manufacturing Equipment Output Prediction</div>', unsafe_allow_html=True)

if page == "🏠 Overview":
    st.markdown('<div class="section-header">📋 Project Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:,}</div>
            <div class="metric-label">Total Records</div>
        </div>
        """.format(len(df)), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.1f}</div>
            <div class="metric-label">Avg Output (parts/hr)</div>
        </div>
        """.format(df['Parts_Per_Hour'].mean()), unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.1f}</div>
            <div class="metric-label">Max Output (parts/hr)</div>
        </div>
        """.format(df['Parts_Per_Hour'].max()), unsafe_allow_html=True)

    with col4:
        if model_data:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.1%}</div>
                <div class="metric-label">Model Accuracy</div>
            </div>
            """.format(model_data['r2_score']), unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <h4>🎯 Problem Statement</h4>
    <p>This application predicts manufacturing equipment output using machine learning to help optimize production efficiency and identify underperforming equipment.</p>

    <h4>📊 Dataset Features</h4>
    <ul>
        <li><b>Injection Temperature:</b> Molten plastic temperature (°C)</li>
        <li><b>Injection Pressure:</b> Hydraulic pressure (bar)</li>
        <li><b>Cycle Time:</b> Time per part cycle (seconds)</li>
        <li><b>Cooling Time:</b> Part cooling duration (seconds)</li>
        <li><b>Material Viscosity:</b> Plastic flow resistance (Pa·s)</li>
        <li><b>Ambient Temperature:</b> Factory temperature (°C)</li>
        <li><b>Machine Age:</b> Equipment age in years</li>
        <li><b>Operator Experience:</b> Experience level (months)</li>
        <li><b>Maintenance Hours:</b> Hours since last maintenance</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif page == "📊 Data Analysis":
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Correlations", "📉 Scatter Plots", "📋 Statistics"])

    with tab1:
        st.subheader("Feature Distributions")

        features = ['Injection_Temperature', 'Injection_Pressure', 'Cycle_Time',
                   'Cooling_Time', 'Material_Viscosity', 'Ambient_Temperature',
                   'Machine_Age', 'Operator_Experience', 'Maintenance_Hours', 'Parts_Per_Hour']

        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        axes = axes.ravel()

        for i, col in enumerate(features):
            if i < len(axes):
                axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7, color='#1f77b4')
                axes[i].set_title(f'Distribution of {col}', fontsize=10)
                axes[i].set_xlabel(col, fontsize=8)
                axes[i].set_ylabel('Frequency', fontsize=8)
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.subheader("Correlation Matrix")

        numerical_features = [
            'Injection_Temperature', 'Injection_Pressure', 'Cycle_Time',
            'Cooling_Time', 'Material_Viscosity', 'Ambient_Temperature',
            'Machine_Age', 'Operator_Experience', 'Maintenance_Hours', 'Parts_Per_Hour'
        ]

        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numerical_features].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', mask=mask, square=True, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        st.pyplot(plt)

        # Top correlations with target
        target_correlations = correlation_matrix['Parts_Per_Hour'].sort_values(ascending=False)
        st.subheader("Top Correlations with Output")
        st.dataframe(target_correlations.head(10).to_frame('Correlation'))

    with tab3:
        st.subheader("Key Feature Relationships")

        key_features = ['Cycle_Time', 'Injection_Temperature', 'Injection_Pressure', 'Cooling_Time']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        for i, feature in enumerate(key_features):
            axes[i].scatter(df[feature], df['Parts_Per_Hour'], alpha=0.6, color='#1f77b4', s=50)
            axes[i].set_xlabel(feature, fontsize=12)
            axes[i].set_ylabel('Parts Per Hour', fontsize=12)
            axes[i].set_title(f'{feature} vs Parts Per Hour', fontsize=14, fontweight='bold')
            axes[i].grid(True, alpha=0.3)

            # Add trend line
            try:
                z = np.polyfit(df[feature], df['Parts_Per_Hour'], 1)
                p = np.poly1d(z)
                x_range = np.linspace(df[feature].min(), df[feature].max(), 100)
                axes[i].plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)
            except:
                pass

        plt.tight_layout()
        st.pyplot(fig)

    with tab4:
        st.subheader("Dataset Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Basic Statistics**")
            st.dataframe(df.describe())

        with col2:
            st.markdown("**Data Types & Missing Values**")
            dtype_df = pd.DataFrame({
                'Data Type': df.dtypes.astype(str),
                'Missing Values': df.isnull().sum(),
                'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(dtype_df.style.set_properties(**{'color': 'black'}))

elif page == "🤖 Model Performance":
    st.markdown('<div class="section-header">🤖 Model Performance Analysis</div>', unsafe_allow_html=True)

    if model_data is None:
        st.error("Model not loaded. Please run the analysis first.")
    else:
        # Load model components
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']
        r2_score_val = model_data['r2_score']

        # Prepare data for evaluation
        df_model = df[features + ['Parts_Per_Hour']].copy()
        df_model = df_model.dropna()

        X = df_model[features]
        y = df_model['Parts_Per_Hour']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = scaler.transform(X_test)

        # Make predictions
        y_test_pred = model.predict(X_test_scaled)

        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("R² Score", f"{r2_score_val:.4f}")
        with col2:
            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            st.metric("RMSE", f"{rmse:.2f}")
        with col3:
            mae = mean_absolute_error(y_test, y_test_pred)
            st.metric("MAE", f"{mae:.2f}")
        with col4:
            mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
            st.metric("MAPE", f"{mape:.1f}%")

        # Predictions vs Actual plot
        st.subheader("Predictions vs Actual Values")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Test set
        ax1.scatter(y_test, y_test_pred, alpha=0.6, color='#1f77b4', s=50)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', alpha=0.8, linewidth=2)
        ax1.set_xlabel('Actual Parts Per Hour', fontsize=12)
        ax1.set_ylabel('Predicted Parts Per Hour', fontsize=12)
        ax1.set_title(f'Test Set: Actual vs Predicted\\nR² = {r2_score_val:.4f}',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Residual plot
        residuals = y_test - y_test_pred
        ax2.scatter(y_test_pred, residuals, alpha=0.6, color='#ff7f0e', s=50)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8, linewidth=2)
        ax2.set_xlabel('Predicted Parts Per Hour', fontsize=12)
        ax2.set_ylabel('Residuals', fontsize=12)
        ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Feature importance
        st.subheader("Feature Importance Analysis")

        feature_importance = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_,
            'Absolute_Coefficient': np.abs(model.coef_)
        })

        feature_importance = feature_importance.sort_values('Absolute_Coefficient', ascending=False)

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(feature_importance['Feature'], feature_importance['Absolute_Coefficient'],
                      color='#1f77b4', alpha=0.8)
        ax.set_xlabel('Absolute Coefficient Value', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Feature Importance in Linear Regression Model', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center', fontsize=10)

        plt.tight_layout()
        st.pyplot(fig)

        # Model coefficients table
        st.subheader("Model Coefficients")
        coeff_df = feature_importance[['Feature', 'Coefficient']].round(4)
        st.dataframe(coeff_df)

elif page == "🔮 Predictions":
    st.markdown('<div class="section-header">🔮 Manufacturing Output Prediction</div>', unsafe_allow_html=True)

    if model_data is None:
        st.error("Model not loaded. Please run the analysis first.")
    else:
        st.markdown("""
        <div class="insight-box">
        Enter the manufacturing parameters below to predict the hourly output (parts per hour).
        The prediction is based on the trained linear regression model.
        </div>
        """, unsafe_allow_html=True)

        # Input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                injection_temp = st.number_input("Injection Temperature (°C)",
                                               value=220.0, min_value=180.0, max_value=250.0, step=0.1)
                injection_press = st.number_input("Injection Pressure (bar)",
                                                value=120.0, min_value=80.0, max_value=150.0, step=0.1)
                cycle_time = st.number_input("Cycle Time (seconds)",
                                           value=25.0, min_value=15.0, max_value=45.0, step=0.1)

            with col2:
                cooling_time = st.number_input("Cooling Time (seconds)",
                                             value=12.0, min_value=8.0, max_value=20.0, step=0.1)
                material_visc = st.number_input("Material Viscosity (Pa·s)",
                                              value=300.0, min_value=100.0, max_value=400.0, step=1.0)
                ambient_temp = st.number_input("Ambient Temperature (°C)",
                                             value=25.0, min_value=18.0, max_value=28.0, step=0.1)

            with col3:
                machine_age = st.number_input("Machine Age (years)",
                                            value=5.0, min_value=1.0, max_value=15.0, step=0.1)
                operator_exp = st.number_input("Operator Experience (months)",
                                             value=60.0, min_value=1.0, max_value=120.0, step=1.0)
                maintenance_hours = st.number_input("Maintenance Hours",
                                                  value=50.0, min_value=0.0, max_value=200.0, step=1.0)

            submitted = st.form_submit_button("🔮 Predict Output")

            if submitted:
                # Prepare input data
                input_data = {
                    "Injection_Temperature": injection_temp,
                    "Injection_Pressure": injection_press,
                    "Cycle_Time": cycle_time,
                    "Cooling_Time": cooling_time,
                    "Material_Viscosity": material_visc,
                    "Ambient_Temperature": ambient_temp,
                    "Machine_Age": machine_age,
                    "Operator_Experience": operator_exp,
                    "Maintenance_Hours": maintenance_hours
                }

                # Make prediction using local model
                input_df = pd.DataFrame([input_data])
                input_scaled = model_data['scaler'].transform(input_df)
                prediction = model_data['model'].predict(input_scaled)[0]
                prediction = max(0, prediction)  # Ensure non-negative

                # Display result
                st.markdown(f"""
                <div class="prediction-result">
                    <h2>🎯 Prediction Result</h2>
                    <div class="prediction-value">{prediction:.1f}</div>
                    <p>Parts Per Hour</p>
                    <p><small>Model R² Score: {model_data['r2_score']:.4f}</small></p>
                </div>
                """, unsafe_allow_html=True)

                # Show input summary
                st.subheader("Input Parameters Summary")
                input_df_display = pd.DataFrame([input_data]).T
                input_df_display.columns = ['Value']
                st.dataframe(input_df_display)

                # Performance interpretation
                st.subheader("Performance Interpretation")

                avg_output = df['Parts_Per_Hour'].mean()
                if prediction > avg_output * 1.1:
                    st.success("🎉 Excellent performance! Output is above average.")
                elif prediction > avg_output * 0.9:
                    st.info("✅ Good performance! Output is within normal range.")
                else:
                    st.warning("⚠️ Below average performance. Consider optimization.")

elif page == "💡 Insights":
    st.markdown('<div class="section-header">💡 Manufacturing Insights & Recommendations</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <h4>🎯 Key Findings from Analysis</h4>
    <ul>
        <li><b>Cycle Time</b> has the strongest impact on output (coefficient: -9.71)</li>
        <li><b>Operator Experience</b> positively affects productivity (coefficient: +3.91)</li>
        <li><b>Injection Temperature</b> optimization improves efficiency (coefficient: +2.02)</li>
        <li>Model achieves <b>83.7% accuracy</b> in predicting manufacturing output</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏭 Optimization Recommendations")

        recommendations = [
            ("🔄 Cycle Time", "Reduce cycle time below 30 seconds for optimal output"),
            ("🌡️ Temperature Control", "Maintain injection temperature in 210-230°C range"),
            ("💪 Pressure Management", "Balance injection pressure with material properties"),
            ("🛠️ Maintenance", "Implement preventive maintenance schedules"),
            ("👥 Operator Training", "Invest in experienced operators (>60 months)"),
            ("📊 Monitoring", "Use predictive analytics for real-time optimization")
        ]

        for icon_text, desc in recommendations:
            st.markdown(f"**{icon_text}**: {desc}")

    with col2:
        st.markdown("### 📈 Business Impact")

        impacts = [
            ("🎯 Efficiency Gain", "10-20% improvement in production efficiency"),
            ("💰 Cost Reduction", "Reduced downtime through predictive maintenance"),
            ("📊 Quality Control", "Better process control and consistency"),
            ("🚀 Scalability", "Data-driven decision making for expansion"),
            ("👥 Workforce", "Optimized training and resource allocation")
        ]

        for icon_text, desc in impacts:
            st.markdown(f"**{icon_text}**: {desc}")

    st.markdown("---")

    # Performance summary
    if model_data:
        st.subheader("📊 Model Performance Summary")

        perf_col1, perf_col2, perf_col3 = st.columns(3)

        with perf_col1:
            st.metric("Model Accuracy", f"{model_data['r2_score']:.1%}")
        with perf_col2:
            st.metric("Dataset Size", f"{len(df):,}")
        with perf_col3:
            st.metric("Features Used", len(model_data['features']))

        st.markdown("""
        <div class="insight-box">
        <h4>🔬 Technical Details</h4>
        <ul>
            <li><b>Algorithm:</b> Linear Regression</li>
            <li><b>Training Method:</b> Scikit-learn with StandardScaler</li>
            <li><b>Validation:</b> 80/20 train-test split</li>
            <li><b>Features:</b> 9 manufacturing parameters</li>
            <li><b>Target:</b> Hourly production output (parts/hour)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🏭 <b>Manufacturing Equipment Output Prediction System</b> | Built with Streamlit & Scikit-learn</p>
    <p><small>Data Science Capstone Project - Real-time Manufacturing Optimization</small></p>
</div>
""", unsafe_allow_html=True)
