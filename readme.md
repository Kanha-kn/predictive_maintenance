# Predictive Maintenance Project

## Project Overview
This project focuses on *predictive maintenance* of industrial machines using sensor data. It combines two key tasks:  
1. *Remaining Useful Life (RUL) Prediction* – Predict how long a machine can operate before failure.  
2. *Failure Classification* – Predict whether a machine will fail (binary classification) and identify the type of failure (multi-class classification).

By integrating these approaches, organizations can plan maintenance proactively, reducing downtime and operational costs.

---

## Problem Statement
Industrial machines are prone to unexpected failures, leading to production losses.  
*Objectives:*  
- Predict if a machine will fail (binary classification).  
- Identify the type of failure (multi-class classification).  
- Predict the Remaining Useful Life (RUL) of machines (regression task).

---

## Datasets

### 1. C-MAPSS Dataset
- *Type:* Time-series data of engine units.  
- *Features:* Sensor readings, operational settings, and RUL labels.  
- *Use Case:* Predict Remaining Useful Life (RUL) of engines.  
- *Reference:* [C-MAPSS Dataset](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Sensor-Data/vrks-gjie)

### 2. AI4I 2020 Predictive Maintenance Dataset
- *Type:* Sensor readings from industrial machines.  
- *Features:* 
  - UDI, Product ID, Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min], Failure Type  
- *Use Case:* 
  - Binary Classification: Predict machine failure (0 = No Failure, 1 = Any Failure)  
  - Multi-class Classification: Predict failure subtype (Heat Dissipation Failure, Power Failure, Overstrain Failure, Tool Wear Failure)  
- *Reference:* [AI4I 2020 Dataset](https://www.kaggle.com/datasets/shivamb/ai4i-2020-predictive-maintenance-dataset)

---

## Project Workflow

### 1. Data Collection
- Load C-MAPSS for RUL prediction.  
- Load AI4I 2020 dataset for failure classification.

### 2. Data Preprocessing
- Handle missing values, anomalies, and outliers.  
- Normalize/scale sensor data.  
- Encode categorical features (e.g., Product ID, Type).  

### 3. Exploratory Data Analysis (EDA)
- Visualize sensor trends and distributions.  
- Analyze failure type frequencies and correlations.  
- Check RUL distributions and trends over engine life cycles.

### 4. Feature Engineering
- Time-series features for RUL prediction (rolling statistics, deltas, etc.).  
- Sensor feature selection for classification tasks.

### 5. Modeling
- *RUL Prediction:* LSTM, GRU, or XGBoost regression.  
- *Binary Classification:* Logistic Regression, Random Forest, XGBoost.  
- *Multi-class Classification:* Random Forest, XGBoost, Neural Networks.

### 6. Evaluation
- RUL Regression Metrics: RMSE, MAE.  
- Binary Classification Metrics: Accuracy, Precision, Recall, F1-score.  
- Multi-class Classification Metrics: Accuracy, Macro/Micro F1-score, Confusion Matrix.

### 7. Prediction & Visualization
- Visualize predicted vs. actual RUL curves.  
- Predict failure probability and subtypes.  
- Plot feature importance to understand key sensors affecting failures.

### 8. Deployment (Optional)
- Create a dashboard to monitor RUL and failure predictions in real-time (using Streamlit or similar tools).

---

## Tools & Libraries
- *Python:* NumPy, Pandas, Matplotlib, Seaborn  
- *Machine Learning:* Scikit-learn, XGBoost  
- *Deep Learning:* TensorFlow, Keras (for LSTM/GRU)  
- *Environment:* Jupyter Notebook / VS Code / Streamlit (for dashboard)

---

## Outcome
- Accurate RUL predictions for engines using C-MAPSS dataset.  
- Binary failure detection and multi-class subtype classification using AI4I dataset.  
- Insights into key sensor features contributing to failures, enabling proactive maintenance.
