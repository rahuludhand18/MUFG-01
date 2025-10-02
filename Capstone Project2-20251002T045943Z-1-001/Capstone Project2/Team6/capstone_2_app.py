import pathlib
import pandas as pd
import streamlit as st
import joblib
import os

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

MODEL_PATH = pathlib.Path(__file__).parent / "models" / "best_model.pkl"
METRICS_PATH = pathlib.Path(__file__).parent / "models" / "model_metrics.csv"

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model at {MODEL_PATH}: {e}")
        return None

model = load_model()

st.title("Heart Disease Prediction")
st.caption("Local inference (no FastAPI call)")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    age = col1.number_input("Age (20-80)", 20, 80, 50)
    sex = col2.selectbox("Sex", [0, 1], format_func=lambda v: "Female" if v == 0 else "Male")
    chest_pain_type = col1.selectbox("Chest Pain Type (1–4)", [1, 2, 3, 4])
    resting_blood_pressure = col2.number_input("Resting BP (80–200)", 80, 200, 120)
    cholesterol = col1.number_input("Cholesterol (100–600)", 100, 600, 200)
    fasting_blood_sugar = col2.selectbox("Fasting Blood Sugar >120 (0/1)", [0, 1])
    resting_ecg = col1.selectbox("Resting ECG (0–2)", [0, 1, 2])
    max_heart_rate = col2.number_input("Max Heart Rate (60–220)", 60, 220, 150)
    exercise_induced_angina = col1.selectbox("Exercise Induced Angina (0/1)", [0, 1])
    st_depression = col2.number_input("ST Depression (0.0–6.0)", 0.0, 6.0, 1.0, step=0.1)
    st_slope = col1.selectbox("ST Slope (1–3)", [1, 2, 3])
    num_major_vessels = col2.selectbox("Num Major Vessels (0–3)", [0, 1, 2, 3])
    thalassemia = col1.selectbox("Thalassemia (3 = normal, 6 = fixed defect, 7 = reversable defect)", [3, 6, 7])
    submitted = st.form_submit_button("Predict")

if submitted:
    if model is None:
        st.error("Model not available.")
    else:
        row = {
            "age": age,
            "sex": sex,
            "cp": chest_pain_type,
            "trestbps": resting_blood_pressure,
            "chol": cholesterol,
            "fbs": fasting_blood_sugar,
            "restecg": resting_ecg,
            "thalach": max_heart_rate,
            "exang": exercise_induced_angina,
            "oldpeak": st_depression,
            "slope": st_slope,
            "ca": num_major_vessels,
            "thal": thalassemia
        }
        df = pd.DataFrame([row])
        try:
            proba = model.predict_proba(df)[:, 1][0]
            pred = 1 if proba >= 0.5 else 0
            st.success(f"Prediction: {'Disease' if pred == 1 else 'No Disease'}")
            st.metric("Risk Score", f"{proba:.3f}")
        except Exception as e:
            st.error(f"Inference error: {e}")

st.markdown("----")
st.markdown("### Model Evaluation & Performance Analysis")

# Show the metrics table in the exact format as your sample image
if os.path.exists(METRICS_PATH):
    metrics_df = pd.read_csv(METRICS_PATH)
    st.dataframe(metrics_df, use_container_width=True)
else:
    st.info("Model metrics will appear here after you train and save the model.")

st.caption("If you prefer using the FastAPI backend, deploy it separately and change this app to call its public URL.")