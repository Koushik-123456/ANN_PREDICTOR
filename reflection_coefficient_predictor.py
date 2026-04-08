"""
reflection_coefficient_predictor.py

Lightweight wrapper that delegates to `predict.py` CLI. This lets you run
the predictor without requiring TensorFlow to be installed in the environment.

Usage:
  python reflection_coefficient_predictor.py --zl 75 --z0 50
  python reflection_coefficient_predictor.py
"""

from predict import main


if __name__ == "__main__":
    main()