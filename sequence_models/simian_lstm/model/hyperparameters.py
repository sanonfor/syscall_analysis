from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from hashlib import sha256
from typing import List, Optional

from tensorflow import keras

from simian_lstm.data.dataset import Dataset
from simian_lstm.data.training_data import TrainingData


@dataclass
class HyperParameters:
    # The experimental settings the parameters were derived from
    experiment: dict
    # The name of the experiment
    tag: Optional[str] = field(default=None)
    # The hashcode of the file the model was loaded from
    file_hashcode: Optional[str] = field(default=None)
    # Use a fixed random seed for building the model (propagates to training)
    fixed_seed: int = field(default=0)
    # Use a fixed random seed for training (but not building the model)
    fixed_seed_train: int = field(default=0)

    # The number of sequences after which we stop loading more files from the data set
    train_size_target: int = field(default=math.inf)
    val_size_target: int = field(default=math.inf)
    max_trace_len: int = field(default=0)
    deduplicate: bool = field(default=False)
    deduplicate_traces: bool = field(default=False)

    # The length of sequences that the model will predict from
    sequence_length: int = field(default=19)

    # The dropout percentage
    dropout: float = field(default=0.0)
    # LSTM-dropout
    lstm_dropout: bool = field(default=True)

    # Add an embedding layer of given size, if nonzero
    # If -1, then embedding size is equal to the size of the first LSTM layer
    embedding: int = field(default=128)

    # The sizes of intermediate LSTM layers
    layers: List[int] = field(default_factory=lambda: [400])

    # Keras parameters
    optimizer: str = field(default="adam")
    initializer: str = field(default="glorot-uniform")
    learning_rate: float = field(default=0.0001)
    clipnorm: float = field(default=None)

    # The number of epochs to train for
    epochs: int = field(default=15)
    # How many epochs to keep training for after val_loss stops improving (0 for unlimited)
    patience: int = field(default=5)

    # If non-empty, filter traces so they only contain listed calls
    allow_calls: List[int] = field(default_factory=list)
    # If non-empty, filter traces so they do not contain listed calls
    deny_calls: List[int] = field(default_factory=list)

    # If using a transformer model, the internal dimension of the dense layer within the transformer
    transformer_dim: int = field(default=256)
    # If using a transformer model, the number of attention heads
    transformer_heads: int = field(default=4)
    # Use a transformer encoder
    transformer_encoder: bool = field(default=True)
    # Use a transformer decoder
    transformer_decoder: bool = field(default=False)
    transformer_layers: int = field(default=1)

    output_sequences: bool = field(default=False)

    @classmethod
    def from_experiment(cls, experiment: dict) -> HyperParameters:
        return HyperParameters(experiment=experiment, **experiment)

    @property
    def param_dict(self) -> dict:
        params = copy.deepcopy(self.__dict__)
        del params["experiment"]
        return params

    @property
    def hashcode(self) -> str:
        if self.file_hashcode:
            return self.file_hashcode
        else:
            return sha256(json.dumps(self.param_dict, sort_keys=True).encode("utf-8")).hexdigest()

    @property
    def optimizer_obj(self):
        if self.optimizer == "adam":
            return keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=self.clipnorm)
        elif self.optimizer == "rmsprop":
            return keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            raise Exception(f"Unknown optimizer {self.optimizer}")

    @property
    def initializer_obj(self):
        if self.initializer == "uniform":
            return keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
        else:
            return keras.initializers.GlorotUniform()

    def load_data(
            self,
            dataset: Dataset,
            **kwargs,
    ) -> TrainingData:
        return dataset.training_data(
            self.sequence_length,
            self.train_size_target,
            self.val_size_target,
            max_len=self.max_trace_len,
            allow_calls=self.allow_calls,
            deny_calls=self.deny_calls,
            one_hot=(self.embedding == 0),
            output_sequences=self.output_sequences,
            deduplicate=self.deduplicate,
            deduplicate_traces=self.deduplicate_traces,
            **kwargs,
        )
