from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import keras
import tensorflow as tf
from keras.utils import set_random_seed

from simian_lstm.data.training_data import TrainingData
from .fit import fit_model
from simian_lstm.model.hyperparameters import HyperParameters


class Model:
    def __init__(
            self,
            model: Optional[keras.Model],
            feature_size: int,
            params: HyperParameters,
            tags: dict,
            model_path: Path = None
    ):
        self._model = model
        self._model_path = model_path
        self.feature_size = feature_size
        self.params = params
        self.experiment = params.experiment
        self.tags = tags
        self.strategy = None

    def __repr__(self):
        return f"Model({self.params.hashcode}, tag={self.params.tag})"

    @property
    def hashcode(self):
        return self.params.hashcode

    @property
    def tag(self):
        return self.params.tag

    @property
    def dataset(self):
        dataset_name = self.stats.get("dataset", {}).get("dataset")
        if dataset_name:
            from simian_lstm.data import Dataset
            return Dataset.load(dataset_name)
        else:
            return None

    @property
    def model(self) -> keras.Model:
        if not self._model:
            self._model = keras.models.load_model(self._model_path)

        return self._model

    @property
    def label(self) -> str:
        return ", ".join(f"{param}: {value}" for param, value in self.experiment.items())

    def _call__(self, x):
        return self.predict(x)

    def predict(self, x, batch_size=128):
        input_shape = self.model.input_shape
        # If this model expects one hot input, we need to convert
        if len(input_shape) == 3:
            feature_size = input_shape[-1]
            x = tf.one_hot(indices=x, depth=feature_size)

        if self.strategy:
            x = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
            x = self.strategy.experimental_distribute_dataset(x)
            return self.model.predict(x, verbose=1)
        else:
            return self.model.predict(x, batch_size=batch_size, verbose=0)

    @property
    def stats(self) -> dict:
        return {
            **self.tags,
            "hashcode": self.params.hashcode,
            "experiment": self.params.experiment,
            "hyperparameters": self.params.param_dict,
            "feature_size": self.feature_size,
        }

    @classmethod
    def build(cls, params: HyperParameters, feature_size: int, version: int):
        if version == 2:
            from simian_lstm.model.lstm import build_model
            return build_model(params, feature_size)
        elif version == 3:
            from simian_lstm.model.transformer import build_model
            return build_model(params, feature_size)
        else:
            raise ValueError(f"Unknown model version {version}")

    @classmethod
    def train(cls, params: HyperParameters, data: TrainingData, version=3):
        if params.fixed_seed:
            set_random_seed(params.fixed_seed)
        else:
            set_random_seed(time.time_ns() % 2**31)
        model = cls.build(params, data.feature_size, version)
        if params.fixed_seed_train:
            set_random_seed(params.fixed_seed_train)
        else:
            set_random_seed(time.time_ns() % 2**31)
        history = fit_model(model, params, data)
        tags = {
            "dataset": data.stats,
            "history": history.history,
            "tag": params.tag,
            "model_version": version,
        }
        return Model(model, data.feature_size, params, tags)

    def save(self, base_path: Path):
        self._model_path = base_path / f"{self.params.hashcode}.model"
        self.model.save(self._model_path)

        self.stats_path(base_path).write_text(json.dumps(self.stats, indent=2))

    def stats_path(self, base_path: Path):
        return base_path / f"{self.params.hashcode}.json"

    def scores_path(self, base_path: Path):
        return base_path / f"{self.params.hashcode}-scores.jsonl"

    @classmethod
    def exists(cls, base_path: Path, hashcode: str) -> bool:
        model_path = base_path / f"{hashcode}.model"
        stats_path = base_path / f"{hashcode}.json"
        return model_path.exists() and stats_path.exists()

    @classmethod
    def load(cls, base_path: Path, hashcode: str):
        model_path = base_path / f"{hashcode}.model"
        stats_path = base_path / f"{hashcode}.json"

        stats: dict = json.loads(stats_path.read_text())
        import inspect
        extra_params = {k: v for k, v in stats["hyperparameters"].items()
                        if k not in inspect.signature(HyperParameters).parameters}
        for k in extra_params:
            del stats["hyperparameters"][k]
        stats["hyperparameters"]["file_hashcode"] = hashcode

        params = HyperParameters(stats["experiment"], **stats["hyperparameters"])
        tags = {k: v for k, v in stats.items() if k in ["dataset", "history"]}

        return Model(None, stats["feature_size"], params, tags, model_path=model_path)

    @classmethod
    def load_tag(cls, base_path: Path, tag: str, strategy) -> List[Model]:
        models = []
        for model_json in base_path.glob("*.json"):
            try:
                j = json.loads(model_json.read_text())
                if j.get("tag", None) == tag:
                    hashcode = model_json.stem
                    models.append(Model.load(base_path, hashcode))
            except json.JSONDecodeError:
                print(f"Error loading model info from {model_json}")

        if not models:
            model = cls.load(base_path, tag)
            if model:
                models.append(model)

        for model in models:
            model.strategy = strategy

        return models

    @classmethod
    def list_tags(cls, base_path: Path) -> List[str]:
        tags = set()

        for model_json in base_path.glob("*.json"):
            j = json.loads(model_json.read_text())
            if j.get("tag"):
                tags.add(j["tag"])

        return list(tags)
