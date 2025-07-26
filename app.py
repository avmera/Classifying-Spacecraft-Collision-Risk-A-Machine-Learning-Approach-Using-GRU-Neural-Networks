import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Masking
import tensorflow as tf
import joblib

@st.cache_data
def load_data():
    df = pd.read_csv("train_data.csv")
    df = df.interpolate()
    return df

FEATURES = [
    'max_risk_scaling', 'time_to_tca', 'mahalanobis_distance', 'max_risk_estimate', 'c_h_per',
    'relative_velocity_t', 'c_recommended_od_span', 'relative_speed', 'c_actual_od_span',
    'c_cd_area_over_mass', 't_j2k_sma', 't_h_per', 'c_ctdot_r', 'c_cr_area_over_mass', 't_h_apo',
    'c_sigma_t', 'c_time_lastob_end', 'c_obs_available', 'c_ctdot_n'
]

def build_model(input_shape):
    model = Sequential([
        Masking(mask_value=0., input_shape=input_shape),
        GRU(100, activation='tanh', return_sequences=True),
        Dropout(0.3),
        GRU(80, activation='tanh', return_sequences=True),
        Dropout(0.2),
        GRU(50, activation='tanh', return_sequences=True),
        Dropout(0.1),
        GRU(30, activation='tanh'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    return model

st.title("Event Risk Classifier and Warning System")

df = load_data()
scaler = joblib.load("scaler.save")

event_col = st.selectbox("Choose the event column to search by:", df.columns)
event_id = st.text_input("Enter Event Value (ID or other unique info):")

if st.button("Check Risk"):
    row = df[df[event_col].astype(str) == str(event_id)]
    if row.empty:
        st.warning("No event found with that value.")
    else:
        X_sample = row[FEATURES].values
        X_scaled = scaler.transform(X_sample)
        X_input = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        model = build_model((1, len(FEATURES)))
        try:
            model.load_weights('gru_risk_model.weights.h5')

        except Exception as e:
            st.error("Could not load trained model weights. Please ensure 'gru_risk_model.h5' is in this directory.")
            st.stop()
        
        y_prob = model.predict(X_input)[0, 0]
        risk_label = "HIGH RISK" if y_prob >= 0.5 else "LOW RISK"
        st.write(f"### Prediction: {risk_label}")
        st.write(f"Probability of High Risk: {y_prob:.2f}")
        if y_prob >= 0.5:
            st.error("⚠️ WARNING: This event is classified as HIGH RISK. Immediate review is recommended!")
