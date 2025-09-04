import streamlit as st
import numpy as np
import pandas as pd
import numpy as np
from tensorflow import keras
import joblib

# Load model and scaler
model = keras.models.load_model("lstm_rul_model.keras")
scaler = joblib.load("scaler_lstm.pkl") # same scaler used in training

st.title("Remaining Useful Life Prediction (LSTM)")
st.write("Upload a CSV file with sensor readings to predict RUL over time")

# list of features (same order as training dataset)
features = ["operational_setting_1","operational_setting_2",
            "sensor_2","sensor_3","sensor_4","sensor_6","sensor_7",
            "sensor_8","sensor_9","sensor_11","sensor_12","sensor_13",
            "sensor_14","sensor_15","sensor_17","sensor_20","sensor_21"
            ]

# file uploader
uploaded_file = st.file_uploader("Upload CSV file with sensor data", type=["csv"])
# window size (same as used in training)
window_size = 30

if uploaded_file is not None:
    # load csv
    df = pd.read_csv(uploaded_file)
    # check columns
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in CSV: {missing_cols}")
    else:
        st.success("CSV loaded successfully")

        # scale features
        scaled_data = scaler.transform(df[features])

        # craete sequence for LSTM
        sequences = []
        for i in range(len(scaled_data)-window_size+1):
            seq = scaled_data[i:i+window_size]
            sequences.append(seq)
        sequences = np.array(sequences)

        # Predict RUL
        preds = model.predict(sequences)
        # Add predictions to dataframe
        df_result = df.iloc[window_size-1:].copy()
        df_result["Predicted_RUL"] = preds.flatten()
        # show result
        st.subheader("Predictions")
        st.dataframe(df_result)

        # Download option
        csv_download = df_result.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV",data=csv_download,file_name="rul_predictions.csv",mime="text/csv")

