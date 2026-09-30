"""Promote a registered model version to Production after an explicit quality gate."""
import os
from mlflow import MlflowClient

MODEL_NAME=os.getenv('MODEL_NAME','LifecycleClassifier')
MIN_AUC=float(os.getenv('MIN_AUC','0.90'))
client=MlflowClient()


def promote(version: str, auc: float):
    if auc < MIN_AUC:
        raise ValueError(f'Quality gate failed: AUC {auc:.4f} < {MIN_AUC:.4f}')
    client.transition_model_version_stage(name=MODEL_NAME, version=version, stage='Production', archive_existing_versions=True)
    return f'{MODEL_NAME} v{version} promoted to Production'
