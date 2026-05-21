from __future__ import annotations

import gc

import keras
from keras.callbacks import EarlyStopping

from simian_lstm.data import TrainingData


def fit_model(model: keras.Model, params, training_data: TrainingData):
    if params.patience:
        callbacks = [EarlyStopping(monitor="val_loss", patience=params.patience, restore_best_weights=True)]
    else:
        callbacks = []

    history = model.fit(
        training_data.training.prefetch(128).cache(),
        epochs=params.epochs,
        shuffle=True,
        callbacks=callbacks,
        validation_data=training_data.validation.prefetch(128).cache(),
    )

    gc.collect()
    import tensorflow
    tensorflow.keras.backend.clear_session()

    return history
