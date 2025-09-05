import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Customer Churn Prediction")
st.write("This simple app uses a Navie Bayes model to predict Customer churn")

churn_df = pd.read_excel("churn_dataset.xlsx")
st.write(churn_df.head())

model = joblib.load("Gaussian_model.pkl")

Age = st.slider("Age", min_value=19, max_value=94, value=20)
Tenure = st.slider("Tenure (months)", min_value=1, max_value=71, value=6)
Gender = st.radio("Gender", ["Male", "Female"], horizontal=True)


gender_num = 1 if Gender == "Male" else 0

if st.button("Predict"):
    input_data = pd.DataFrame([[Age, Tenure, gender_num]], 
                              columns=["Age", "Tenure", "Sex"])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is likely to stay.")
    
    st.write(f"Probability of Churn: {probability:.2f}")