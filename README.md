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


## Deploying to Render.com (one-click style)

You can deploy the Streamlit web UI to Render.com and get a public URL. Steps:

1. Push your repo to GitHub (already done for this project).
2. Sign in to https://render.com and choose "New" → "Web Service".
3. Connect your GitHub account and choose this repository (`ANN_PREDICTOR`) and branch `main`.
4. For Environment choose **Docker** (we included a `Dockerfile`).
5. Leave the Build Command blank. For Start Command use:

```
streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```

6. Create the service. Render will build the container and assign a public URL like `https://<your-service-name>.onrender.com`.

Notes:
- The Docker image installs Python 3.10 and the packages in `requirements.txt`.
- If you prefer not to use Docker, you can select the Python environment option in Render and set the Start Command above; Render will install dependencies from `requirements.txt`.
- If TensorFlow wheel installation fails on Render's builder for your selected plan, consider using a smaller model or a CPU-only compatible wheel; alternatively use the included scikit-learn fallback (already trained model is checked into `models/`).
## Performance and evaluation

The `train.py` script prints MSE/MAE on the test set and also reports the fraction of predictions within absolute tolerances (e.g., 0.01 and 0.05). Plots are saved to `outputs/pred_vs_actual.png` and `outputs/training_history.png`.

## Notes & next steps

- The project currently uses real-valued impedances (real Γ). For complex impedances you can modify `utils.generate_dataset` to sample complex values and keep the complex `Γ`.
- The ANN architecture is intentionally simple; you can tune depth, width, regularization, or try other models for improved performance.

If you want, I can now train the model here (if you want me to attempt training in this environment), or run a small smoke test to verify the scripts execute. Which would you like next?

## Target Python Version

This project targets Python 3.10 to ensure compatibility with TensorFlow and other binary packages.

Suggested options to get a Python 3.10 environment on Windows:

- Install from the official installer: https://www.python.org/downloads/release/python-31013/
- Use `winget` (Windows 10/11):

```powershell
winget install --id=Python.Python.3.10 -e --source winget
```

- Use `conda` (Anaconda/Miniconda):

```bash
conda create -n ann-ref python=3.10 pip -y
conda activate ann-ref
pip install -r requirements.txt
```

After installing Python 3.10, create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate  # PowerShell/CMD on Windows
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CI and development tools use Python 3.10 as well (see `.python-version`, `Pipfile`, and `pyproject.toml`).