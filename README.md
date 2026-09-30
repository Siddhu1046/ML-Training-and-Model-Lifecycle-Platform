# ML Training and Model Lifecycle Platform

MLOps project demonstrating dataset versioning, experiment tracking, model evaluation, registry-style promotion, and reproducible training.

## Stack
Python, MLflow, DVC, Git, scikit-learn, Docker, FastAPI.

## Lifecycle
Data version -> train -> evaluate -> log metrics/artifacts -> register candidate -> approval -> production alias -> rollback.