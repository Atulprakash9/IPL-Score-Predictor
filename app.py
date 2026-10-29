import streamlit as st
from pathlib import Path
import joblib
import numpy as np
from tensorflow import keras

ROOT = Path(__file__).resolve().parent

st.title("IPL Score Predictor")
st.write("Predict final first-innings score from current match situation.")

@st.cache_resource
def load_model_and_scaler():
    model_path = ROOT / 'models' / 'ipl_ann.keras'
    scaler_path = ROOT / 'models' / 'scaler.joblib'
    if not model_path.exists() or not scaler_path.exists():
        return None, None
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error("Model ya Scaler file nahi mili! Check karein ki 'models' folder me ipl_ann.keras aur scaler.joblib मौजूद hain.")
else:
    col1, col2 = st.columns(2)
    with col1:
        overs = st.number_input('Completed Overs (0 to 19)', min_value=0, max_value=19, value=5, step=1)
    with col2:
        balls = st.number_input('Balls in current over (0 to 5)', min_value=0, max_value=5, value=0, step=1)

    col3, col4 = st.columns(2)
    with col3:
        runs = st.number_input('Current Runs', min_value=0, max_value=350, value=40, step=1)
    with col4:
        wickets = st.number_input('Wickets Fallen (0 to 9)', min_value=0, max_value=9, value=1, step=1)

    if st.button('Predict Final Score'):
        elapsed = overs + (balls / 6.0)

        if elapsed <= 0:
            st.warning("Match shuru hona chahiye (overs ya balls 0 se zyada honi chahiye).")
        else:
            run_rate = runs / elapsed
            x = np.array([[elapsed, runs, wickets, run_rate]], dtype=np.float32)
            scaled_x = scaler.transform(x)
            prediction = float(model.predict(scaled_x, verbose=0)[0][0])
            final_score = int(round(max(runs, prediction)))

            st.success(f"Predicted Final Score: {final_score} runs")