"""
A minimal Streamlit app for interactive prediction (optional).
Run with: `streamlit run streamlit_app.py`
"""

try:
    import streamlit as st
    import joblib
    import numpy as np
    from utils import reflection_coefficient
    TF_AVAILABLE = True
    try:
        from tensorflow.keras.models import load_model
    except Exception:
        TF_AVAILABLE = False
except Exception:
    raise SystemExit('Streamlit or other dependencies are not installed. Install with `pip install streamlit`')

st.title('Reflection Coefficient Predictor (ANN)')

zl = st.number_input('Load Impedance Z_L (ohms)', min_value=0.0, value=75.0, format='%f')
z0 = st.number_input('Characteristic Impedance Z_0 (ohms)', min_value=0.0, value=50.0, format='%f')

if st.button('Predict'):
    gamma_true = reflection_coefficient(zl, z0)
    st.write('Analytic Γ =', gamma_true)
    model_path = 'models/ann_reflection.h5'
    scaler_path = 'models/scaler.joblib'
    if TF_AVAILABLE and os.path.exists(model_path) and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        model = load_model(model_path)
        Xs = scaler.transform([[zl, z0]])
        pred = model.predict(Xs).squeeze()
        st.write('ANN predicted Γ =', float(pred))
    else:
        st.info('ANN model or scaler not available. Train with `python train.py`.')
