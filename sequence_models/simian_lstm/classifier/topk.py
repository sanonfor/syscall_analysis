import numpy as np
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view

from simian_lstm.classifier import ModelClassifier
from simian_lstm.classifier.model import DEFAULT_BATCH_SIZE
from simian_lstm.data import SyscallTrace
from simian_lstm.model import Model


class TopKClassifier(ModelClassifier):
    def __init__(self, model: Model, k: int, **kwargs):
        super().__init__(model, **kwargs)
        self.k = k

    def hash_data(self) -> dict:
        return {
            **super().hash_data(),
            "top_k": self.k,
        }

    def likelihood(self, trace: SyscallTrace, batch_size=DEFAULT_BATCH_SIZE) -> float:
        """Returns the proportion of calls in the trace that are in the top K most likely."""
        if len(trace) < 2:
            return 0.0

        return self.passing_calls(trace, batch_size) / len(trace)

    def passing_calls(self, trace: SyscallTrace, batch_size=DEFAULT_BATCH_SIZE) -> int:
        passes = 0

        subseqs = sliding_window_view(trace.trace[:-1], self.sequence_length)
        actuals = trace.trace[self.sequence_length:]
        actuals[actuals >= self.model.feature_size] = 0

        splits = max(len(subseqs) / batch_size, 1)
        subseq_partitions = np.array_split(subseqs, splits)
        actual_partitions = np.array_split(actuals, splits)

        for sp, ap in zip(subseq_partitions, actual_partitions):
            sp = tf.convert_to_tensor(sp)
            predictions: tf.Tensor = self.model(sp)

            for prediction, actual in zip(predictions, ap):
                if self.check_call(prediction, actual):
                    passes += 1

        return passes

    def check_call(self, prediction, actual) -> bool:
        # refer to https://stackoverflow.com/a/23734295
        top_k = np.argpartition(prediction.numpy(), -self.k)[-self.k:]
        return actual in top_k
