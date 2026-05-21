from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import tensorflow as tf


@dataclass
class TrainingData:
    training: tf.data.Dataset
    validation: tf.data.Dataset
    feature_size: int
    dataset: Optional[str] = field(default=None)
    meta: Optional[dict] = field(default=None)

    @property
    def stats(self):
        return {
            "dataset": self.dataset,
            "feature_size": self.feature_size,
            "train_size": len(self.training),
            "val_size": len(self.validation),
            "meta": self.meta,
        }

    def distribute(self, strategy):
        self.training = strategy.experimental_distribute_dataset(self.training)
        self.validation = strategy.experimental_distribute_dataset(self.validation)
