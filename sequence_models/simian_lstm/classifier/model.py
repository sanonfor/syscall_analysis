import gc
import math
import sys
from typing import List, Tuple

import numpy as np
import tensorflow as tf
# noinspection PyProtectedMember
from numpy.lib.stride_tricks import sliding_window_view

from simian_lstm.data import SyscallTrace, Dataset
from simian_lstm.model import Model

from .classifier import Classifier


DEFAULT_BATCH_SIZE = 64


class ModelClassifier(Classifier):
    def __init__(self, model: Model, sequence_length=None, batch_size=None, label=None, **_kwargs):
        super().__init__()
        self.model = model
        self.label = label
        self.sequence_length = sequence_length or model.params.sequence_length
        self.batch_size = batch_size or DEFAULT_BATCH_SIZE

    @property
    def name(self) -> str:
        return self.model.params.tag

    @property
    def dataset(self) -> Dataset:
        return self.model.dataset

    @property
    def feature_size(self) -> int:
        return self.model.feature_size

    def hash_data(self) -> dict:
        return {
            **super().hash_data(),
            "model": self.model.params.hashcode,
            "tag": self.model.params.tag,
            "sequence_length": self.sequence_length,
        }

    def likelihood(self, trace: SyscallTrace) -> float:
        """Returns the average negative log likelihood of a sequence."""
        if len(trace) < 2:
            return 0.0

        steps = self.step_likelihoods(trace)
        return sum(steps) / len(steps)

    def likelihood_multi(self, traces: List[SyscallTrace]) -> List[float]:
        seqs, actual = self.split_trace(traces[0])
        ranges = [range(len(actual))]

        for trace in traces[1:]:
            s, a = self.split_trace(trace)
            seqs = np.concatenate(seqs, s)
            actual = np.concatenate(actual, a)
            latest = ranges[-1].stop
            ranges.append(range(latest, latest+len(a)))

        predictions = self.model.predict(seqs, self.batch_size)
        if len(predictions.shape) == 3:
            predictions = predictions[:, -1, :]

        steps = [self.neg_log_likelihood(prediction[a])
                 for prediction, a in zip(predictions, actual)]

        return [np.mean(steps[r.start:r.stop]) for r in ranges]

    def split_trace(self, trace: SyscallTrace) -> Tuple[np.ndarray, np.ndarray]:
        subseqs = sliding_window_view(trace.trace[:-1], self.sequence_length)
        actuals: np.ndarray = trace.trace[self.sequence_length:]
        actuals[actuals >= self.model.feature_size] = 0
        return subseqs, actuals

    def step_likelihoods(self, trace: SyscallTrace):
        subseqs, actuals = self.split_trace(trace)

        predictions = self.model.predict(subseqs, self.batch_size)
        if len(predictions.shape) == 3:
            predictions = predictions[:, -1, :]

        steps = [self.neg_log_likelihood(prediction[actual])
                 for prediction, actual in zip(predictions, actuals)]

        gc.collect()
        tf.keras.backend.clear_session()
        return steps

    @staticmethod
    def neg_log_likelihood(probability: float) -> float:
        if probability > 0:
            return -math.log(probability)
        else:
            return sys.float_info.max / 100000

    def can_classify(self, trace: SyscallTrace) -> bool:
        return len(trace) > self.sequence_length + 1
