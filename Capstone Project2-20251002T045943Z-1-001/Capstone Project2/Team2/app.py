import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open("heart_disease_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Heart Disease Prediction App 🫀")

st.write("Enter patient details to predict the likelihood of heart disease.")

# Input features (replace with your dataset's actual features)
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (0=Female, 1=Male)", [0,1])
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=0)
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0=No, 1=Yes)", [0,1])
restecg = st.number_input("Resting ECG (0-2)", min_value=0, max_value=2, value=0)
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina (0=No, 1=Yes)", [0,1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.number_input("Slope of Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, value=1)
ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
thal = st.number_input("Thalassemia (1=Normal, 2=Fixed, 3=Reversible)", min_value=1, max_value=3, value=2)

# Collect inputs into an array
input_features = np.array([age, sex, cp, trestbps, chol, fbs, restecg,
                           thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]

    if prediction == 1:
        st.error(f"⚠️ The patient is likely to have heart disease. Probability: {probability:.2f}")
    else:
        st.success(f"✅ The patient is unlikely to have heart disease. Probability: {probability:.2f}")
