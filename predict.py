"""
predict.py

Simple CLI for loading the trained ANN and predicting the reflection coefficient
for given Z_L and Z_0 values. If a trained model is not found, the script will
fall back to the analytic formula.

Usage:
  python predict.py --zl 75 --z0 50
  python predict.py        # prompts for input
"""

import os
import argparse
import joblib
import numpy as np

from utils import reflection_coefficient

try:
    # lazy import so the module can be executed even without TF installed
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


def predict_with_model(zl: float, z0: float) -> float:
    # Try Keras model first
    keras_model_path = 'models/ann_reflection.h5'
    scaler_path = 'models/scaler.joblib'
    sklearn_model_path = 'models/sklearn_mlp.joblib'

    X = np.array([[zl, z0]], dtype=float)

    # If scaler missing, raise
    if not os.path.exists(scaler_path):
        raise FileNotFoundError('Scaler not found in models/.')
    scaler = joblib.load(scaler_path)
    Xs = scaler.transform(X)

    # Prefer Keras model if available
    if os.path.exists(keras_model_path) and TF_AVAILABLE:
        model = load_model(keras_model_path)
        pred = model.predict(Xs).squeeze()
        return float(pred)

    # Fall back to sklearn model if present
    if os.path.exists(sklearn_model_path):
        skl = joblib.load(sklearn_model_path)
        pred = skl.predict(Xs).squeeze()
        return float(pred)

    raise FileNotFoundError('No trained model found in models/.')


def main():
    parser = argparse.ArgumentParser(description='Predict reflection coefficient (Γ)')
    parser.add_argument('--zl', type=float, help='Load impedance (ohms)')
    parser.add_argument('--z0', type=float, help='Characteristic impedance (ohms)')
    args = parser.parse_args()

    if args.zl is None or args.z0 is None:
        try:
            args.zl = float(input('Enter Load Impedance Z_L (ohms): '))
            args.z0 = float(input('Enter Characteristic Impedance Z_0 (ohms): '))
        except ValueError:
            print('Invalid input. Please enter numeric values.')
            return

    zl = args.zl
    z0 = args.z0

    # Analytic value
    gamma_true = reflection_coefficient(zl, z0)

    # ANN prediction (if available)
    gamma_pred = None
    try:
        gamma_pred = predict_with_model(zl, z0)
    except Exception:
        gamma_pred = None

    print('\nResults:')
    print(f'  Analytic Γ = (Z_L - Z_0)/(Z_L + Z_0) = {gamma_true}')
    if gamma_pred is not None:
        print(f'  ANN predicted Γ = {gamma_pred:.6f}')
        print(f'  Absolute error = {abs(float(gamma_true) - float(gamma_pred)):.6e}')
    else:
        print('  ANN model not available. To train a model, run `python train.py`.')


if __name__ == '__main__':
    main()
