from pathlib import Path
from typing import Optional

from simian_lstm.data.dataset import Dataset, TracePredicate
from simian_lstm.model.hyperparameters import HyperParameters
from simian_lstm.model.model import Model
from simian_lstm.util import experiment_permutations

import tensorflow as tf


def train(
        dataset: Dataset,
        experiment_params: dict,
        save_path: Path,
        tag: Optional[str] = None,
        predicate: TracePredicate = lambda _: True,
        model_version: int = 2,
        strategy: Optional[tf.distribute.Strategy] = None,
):
    for k, v in experiment_params.items():
        if not isinstance(v, list):
            experiment_params[k] = [v]

    desired_experiments = experiment_permutations(experiment_params)

    models = []
    for x in desired_experiments:
        params = HyperParameters(x, tag=tag, **x)
        if Model.exists(save_path, params.hashcode):
            continue

        print(f"Running experiment {x}")
        training_data = params.load_data(dataset, predicate=predicate)
        if strategy:
            training_data.distribute(strategy)
        print(f"Training on {len(training_data.training)} Samples")
        model = Model.train(params, training_data, version=model_version)
        model.save(save_path)
        models.append(model)

    return models


def tune(dataset: Dataset):
    from simian_lstm.model.tune import build_tuner, search
    tuner = build_tuner(dataset, max_trials=20)
    tuner.search_space_summary()
    search(tuner, dataset)
