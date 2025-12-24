"""Learner package - model training and prediction."""

from learner.trainer import train_model
from learner.predictor import predict

__all__ = [
    "train_model",
    "predict",
]
