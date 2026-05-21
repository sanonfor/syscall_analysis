from pathlib import Path

import keras_tuner as kt
import tensorflow as tf

from simian_lstm.data.dataset import Dataset
from simian_lstm.model import Model
from simian_lstm.model.hyperparameters import HyperParameters


def build_hyperparams(hp: kt.HyperParameters, sequence_length: int) -> HyperParameters:
    layer_count = hp.Int("layer_count", min_value=1, max_value=4)
    layer_size = hp.Int("layer_size", min_value=128, max_value=1024)
    dropout = hp.Float("dropout", min_value=0.0, max_value=0.5)
    learning_rate= hp.Float("learning_rate", min_value=0.00001, max_value=0.0001, sampling="log")
    embedding = hp.Int("embedding", min_value=32, max_value=512)

    return HyperParameters(
        hp.values,
        layers=[layer_size]*layer_count,
        sequence_length=sequence_length,
        embedding=embedding,
        learning_rate=learning_rate,
        optimizer="adam",
    )


def hypermodel(feature_size: int, sequence_length: int = 19, builder=build_hyperparams):
    def _hypermodel(hp: kt.HyperParameters):
        params = builder(hp, sequence_length=sequence_length)
        return Model.build(params, feature_size, version=2)
    return _hypermodel


def build_tuner(
        dataset: Dataset,
        max_trials: int = 50,
        objective="val_loss",
        executions_per_trial=5,
        overwrite=False,
        directory="tuning",
        hyper=hypermodel,
):
    return kt.BayesianOptimization(
        hypermodel=hyper(dataset.feature_size),
        objective=objective,
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        overwrite=overwrite,
        directory=directory,
        project_name=dataset.name,
    )


def search(
        tuner,
        dataset: Dataset,
        epochs=50,
        patience=5,
):
    dummy_params = HyperParameters({})
    data = dummy_params.load_data(dataset)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor=tuner.oracle.objective.name, patience=patience)
    tuner.search(
        data.training.prefetch(128).cache(),
        epochs=epochs,
        shuffle=True,
        validation_data=data.validation.prefetch(128).cache(),
        callbacks=[stop_early],
    )

    return tuner


def save_best(base_path: Path, dataset: Dataset, tuner, tag: str, n=1, sequence_length: int = 19):
    keras_models = tuner.get_best_models(n)
    keras_hps = tuner.get_best_hyperparameters(n)
    local_hps = [build_hyperparams(hp, sequence_length) for hp in keras_hps]
    for hp in local_hps:
        hp.tag = tag
    tags = {"tag": tag, "history": {k: [] for k in ["val_accuracy", "training_accuracy", "val_loss", "training_loss"]}}
    local_models = [Model(km, dataset.feature_size, hp, tags) for km, hp in zip(keras_models, local_hps)]
    for model in local_models:
        model.save(base_path)
