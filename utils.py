"""
utils.py

Utility functions for the Reflection Coefficient ANN project.

Provides:
- reflection_coefficient: compute Γ = (Z_L - Z_0) / (Z_L + Z_0)
- generate_dataset: random dataset generator returning a pandas DataFrame

Supports scalar, vector, and complex-valued impedances.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def reflection_coefficient(Z_L, Z_0):
    """
    Compute the reflection coefficient Γ = (Z_L - Z_0) / (Z_L + Z_0).

    Parameters
    - Z_L, Z_0 : scalar, list, or numpy array. Can be real or complex.

    Returns
    - numpy array or scalar with the same shape as inputs (complex dtype).

    Notes
    - Avoids division-by-zero by adding a tiny epsilon where Z_L + Z_0 == 0.
    """
    Z_L_arr = np.array(Z_L, dtype=np.complex128)
    Z_0_arr = np.array(Z_0, dtype=np.complex128)
    denom = Z_L_arr + Z_0_arr
    # Avoid exact zero denominator
    eps = 1e-12
    denom = np.where(denom == 0, denom + eps, denom)
    gamma = (Z_L_arr - Z_0_arr) / denom
    return gamma


def generate_dataset(n_samples: int = 10000,
                     z_min: float = 1.0,
                     z_max: float = 100.0,
                     seed: int = 42) -> pd.DataFrame:
    """
    Generate a dataset of random impedances and compute the corresponding reflection coefficients.

    Returns a DataFrame with columns: ['Z_L', 'Z_0', 'Gamma']
    """
    rng = np.random.default_rng(seed)
    Z_L = rng.uniform(z_min, z_max, size=n_samples)
    Z_0 = rng.uniform(z_min, z_max, size=n_samples)
    Gamma = reflection_coefficient(Z_L, Z_0).real  # real-valued dataset by default
    df = pd.DataFrame({
        'Z_L': Z_L,
        'Z_0': Z_0,
        'Gamma': Gamma
    })
    return df


if __name__ == "__main__":
    # Quick smoke-run when executed directly
    print("Example: Γ for Z_L=75, Z_0=50 ->", reflection_coefficient(75, 50))
