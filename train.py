"""
train.py

Train an ANN to predict the reflection coefficient Γ from Z_L and Z_0.

Usage:
  python train.py            # runs with default settings
  python train.py --samples 20000 --epochs 100

Outputs:
- saved Keras model: models/ann_reflection.h5
- saved scaler: models/scaler.joblib
- saved plots in outputs/
- dataset saved to data/dataset.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    callbacks = None
    print('TensorFlow not available; will use scikit-learn MLP as a fallback.')

from utils import generate_dataset, reflection_coefficient


def build_model(input_dim=2, lr=1e-3):
    if not TF_AVAILABLE:
        raise RuntimeError('TensorFlow is not available')
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='linear')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='mse', metrics=['mae'])
    return model


def build_sklearn_mlp(hidden_layer_sizes=(128, 64, 32), lr=1e-3, max_iter=200, batch_size=32):
    from sklearn.neural_network import MLPRegressor
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                       activation='relu',
                       solver='adam',
                       learning_rate_init=lr,
                       max_iter=max_iter,
                       batch_size=batch_size,
                       random_state=42)
    return mlp


def compute_accuracy_within_tolerance(y_true, y_pred, tol=0.01):
    """Return fraction of predictions within absolute tolerance tol."""
    return float(np.mean(np.abs(y_true - y_pred) <= tol))


def plot_pred_vs_actual(y_true, y_pred, out_path):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel('Actual Γ')
    plt.ylabel('Predicted Γ')
    plt.title('Predicted vs Actual Reflection Coefficient')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_history(history, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(args):
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    print('Generating dataset...')
    df = generate_dataset(n_samples=args.samples, z_min=args.z_min, z_max=args.z_max, seed=args.seed)
    df.to_csv('data/dataset.csv', index=False)
    print('Dataset saved to data/dataset.csv')

    X = df[['Z_L', 'Z_0']].values
    y = df['Gamma'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, 'models/scaler.joblib')
    print('Scaler saved to models/scaler.joblib')

    np.random.seed(args.seed)

    if TF_AVAILABLE:
        tf.random.set_seed(args.seed)
        model = build_model(input_dim=2, lr=args.lr)
        model.summary()

        cb = [
            callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            callbacks.ModelCheckpoint('models/ann_reflection_best.h5', save_best_only=True, monitor='val_loss')
        ]

        history = model.fit(X_train_scaled, y_train,
                            validation_split=0.2,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            callbacks=cb,
                            verbose=1)

        # Save final model (weights already restored by EarlyStopping if used)
        model.save('models/ann_reflection.h5')
        print('Keras model saved to models/ann_reflection.h5')

        # Evaluate
        loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
        y_pred = model.predict(X_test_scaled).squeeze()

        mse = mean_squared_error(y_test, y_pred)
        mae_sklearn = mean_absolute_error(y_test, y_pred)

        acc_0_01 = compute_accuracy_within_tolerance(y_test, y_pred, tol=0.01)
        acc_0_05 = compute_accuracy_within_tolerance(y_test, y_pred, tol=0.05)

        print(f"Test MSE (Keras): {loss:.6f}")
        print(f"Test MAE (Keras): {mae:.6f}")
        print(f"Sklearn MSE: {mse:.6f}, Sklearn MAE: {mae_sklearn:.6f}")
        print(f"Accuracy within 0.01: {acc_0_01*100:.2f}%")
        print(f"Accuracy within 0.05: {acc_0_05*100:.2f}%")

        # Save small sample of predictions
        sample_df = pd.DataFrame({
            'Z_L': X_test[:, 0],
            'Z_0': X_test[:, 1],
            'Gamma_actual': y_test,
            'Gamma_pred': y_pred,
            'abs_error': np.abs(y_test - y_pred)
        })
        sample_df.sort_values('abs_error', inplace=True)
        sample_df.to_csv('outputs/sample_predictions.csv', index=False)
        print('Sample predictions saved to outputs/sample_predictions.csv')

        # Plots
        plot_pred_vs_actual(y_test, y_pred, 'outputs/pred_vs_actual.png')
        plot_history(history, 'outputs/training_history.png')
        print('Plots saved to outputs/')

    else:
        # Fallback: train a scikit-learn MLPRegressor
        print('Training scikit-learn MLPRegressor as fallback...')
        mlp = build_sklearn_mlp(max_iter=args.epochs, lr=args.lr, batch_size=args.batch_size)
        mlp.fit(X_train_scaled, y_train)
        # Save sklearn model
        joblib.dump(mlp, 'models/sklearn_mlp.joblib')
        print('Sklearn MLP model saved to models/sklearn_mlp.joblib')

        y_pred = mlp.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mae_sklearn = mean_absolute_error(y_test, y_pred)

        acc_0_01 = compute_accuracy_within_tolerance(y_test, y_pred, tol=0.01)
        acc_0_05 = compute_accuracy_within_tolerance(y_test, y_pred, tol=0.05)

        print(f"Sklearn MSE: {mse:.6f}, Sklearn MAE: {mae_sklearn:.6f}")
        print(f"Accuracy within 0.01: {acc_0_01*100:.2f}%")
        print(f"Accuracy within 0.05: {acc_0_05*100:.2f}%")

        # Save small sample of predictions
        sample_df = pd.DataFrame({
            'Z_L': X_test[:, 0],
            'Z_0': X_test[:, 1],
            'Gamma_actual': y_test,
            'Gamma_pred': y_pred,
            'abs_error': np.abs(y_test - y_pred)
        })
        sample_df.sort_values('abs_error', inplace=True)
        sample_df.to_csv('outputs/sample_predictions.csv', index=False)
        print('Sample predictions saved to outputs/sample_predictions.csv')

        # Plots
        plot_pred_vs_actual(y_test, y_pred, 'outputs/pred_vs_actual.png')
        # If sklearn recorded a loss curve, plot it
        if hasattr(mlp, 'loss_curve_'):
            plt.figure(figsize=(8, 5))
            plt.plot(mlp.loss_curve_, label='train loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Sklearn MLP Training Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('outputs/training_history.png')
            plt.close()
        print('Plots saved to outputs/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ANN to predict reflection coefficient')
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--z_min', type=float, default=1.0)
    parser.add_argument('--z_max', type=float, default=100.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
