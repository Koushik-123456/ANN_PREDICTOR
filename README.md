# Reflection Coefficient Prediction using ANN

This repository contains a complete Python project that trains an Artificial Neural Network (ANN) to predict the reflection coefficient (Γ) of a transmission line given the load impedance `Z_L` and the characteristic impedance `Z_0`.

**Mathematical formula**

The reflection coefficient is defined as:

$$\Gamma = \frac{Z_L - Z_0}{Z_L + Z_0}$$

Where `Z_L` is the load impedance and `Z_0` the characteristic impedance. This project uses real-valued impedances by default.

**Files added**
- [utils.py](utils.py): utilities for the analytic formula and dataset generation.
- [train.py](train.py): script to generate data, train the ANN, evaluate and save results.
- [predict.py](predict.py): CLI for predicting Γ from input impedances (loads the trained model).
- [streamlit_app.py](streamlit_app.py): optional lightweight web UI (run with `streamlit run streamlit_app.py`).
- [requirements.txt](requirements.txt): Python dependencies.
- `models/` directory: model and scaler saved after training.
- `data/` and `outputs/` directories: dataset and plots.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model (creates `models/ann_reflection.h5` and `models/scaler.joblib`):

```bash
python train.py --samples 10000 --epochs 100
```

3. Predict from the command line:

```bash
python predict.py --zl 75 --z0 50
```

4. (Optional) Run the Streamlit UI:

```bash
streamlit run streamlit_app.py
```

## What the scripts do

- `utils.py`: Implements the analytic formula and a dataset generator using random `Z_L` and `Z_0` samples.
- `train.py`: Generates data, scales features, trains a Keras model, evaluates on a hold-out test set, saves the trained model and scaler, and writes plots to `outputs/`.
- `predict.py`: Loads the scaler and trained model (if present) and returns the ANN prediction; otherwise reports the analytic value.

## Performance and evaluation

The `train.py` script prints MSE/MAE on the test set and also reports the fraction of predictions within absolute tolerances (e.g., 0.01 and 0.05). Plots are saved to `outputs/pred_vs_actual.png` and `outputs/training_history.png`.

## Notes & next steps

- The project currently uses real-valued impedances (real Γ). For complex impedances you can modify `utils.generate_dataset` to sample complex values and keep the complex `Γ`.
- The ANN architecture is intentionally simple; you can tune depth, width, regularization, or try other models for improved performance.

If you want, I can now train the model here (if you want me to attempt training in this environment), or run a small smoke test to verify the scripts execute. Which would you like next?