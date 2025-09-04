# imports
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Binary XGBoost model
binary_model = joblib.load("xgb_model.pkl")
binary_scaler = joblib.load("scaler.pkl")

# Multi-output XGboost model
multi_model = joblib.load("xgb_model.pkl")
multi_scaler = joblib.load("scaler_multi.pkl")

# Application title and description
st.title("Predictive Maintenance - Machine Failure Prediction")
st.markdown("""
This app predicts whether a machine will fail or keep running.
if it predicts failure, it also predcits the type(s) of failure (TWF, HDF, OSF, PWF, RNF)based 
on operating conditions like temperature, speed, torque, and tool wear""")

# user inputs
machine_type = st.selectbox("Machine Type",["L","M","H"])
air_temp = st.number_input("Air Temperature [K]", 295.0, 320.0, 298.1)
process_temp = st.number_input("Process Temperature [K]", 305.0, 330.0, 308.6)
rot_speed = st.number_input("Rotational Speed [rpm]", 1200, 3000, 1551)
torque = st.number_input("Torque [nm]", 0.0, 100.0, 42.8)
tool_wear = st.number_input("Tool Wear [min]", 0, 250, 10)

# Encode machine type
if machine_type == "L":
    type_L,type_M = 1,0
elif machine_type == "M":
    type_L,type_M = 0,1
else:
    type_L,type_M = 0,0

# ------------------------------
# Arrange input arrays
# ------------------------------
input_features = np.array([[air_temp, process_temp, rot_speed, torque, tool_wear, type_L, type_M]])

# Scale inputs for binary and multi-output models
input_binary_scaled = binary_scaler.transform(input_features)
input_multi_scaled = multi_scaler.transform(input_features)

# ------------------------------
# Prediction button
# ------------------------------
if st.button("Predict Machine Failure"):

    # Step 1: Binary prediction
    binary_pred = binary_model.predict(input_binary_scaled)[0]
    binary_prob = binary_model.predict_proba(input_binary_scaled)[0][1]

    if binary_pred == 0:
        st.success(f"Machine is safe (failure probability: {binary_prob:.2f})")
    else:
        st.error(f"Machine may fail (failure probability: {binary_prob:.2f})")

        # Step 2: Multi-output prediction
        subtype_pred = multi_model.predict(input_multi_scaled)
        if subtype_pred.ndim > 1:
            subtype_pred = subtype_pred[0]
        probs_list = multi_model.predict_proba(input_multi_scaled)
        subtype_prob = [p[0][1] if p.ndim> 1 else  p[1] for p in probs_list]

        # Define failure subtypes
        subtypes = ["TWF","HDF","PWF","OSF","RNF"]

        # Filter predicted subtypes
        predicted_subtypes = {sub: prob for sub, prob, flag in zip(subtypes, subtype_prob, subtype_pred) if flag==1}

        if predicted_subtypes:
            st.subheader("Predicted Failure Subtypes:")
            for sub, prob in predicted_subtypes.items():
                st.write(f"{sub} (probability: {prob:.2f})")
        else:
            st.write("No specific subtype predicted, but machine may fail")
