import gc
import math
import sys
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from simian_lstm.classifier import Classifier
from simian_lstm.data import SyscallTrace, Dataset
from simian_lstm.model.model import Model


class OldClassifier(Classifier):
    def __init__(self, model: Model, window_size=10):
        super().__init__()
        self.model = model
        self.window_size = window_size

    @property
    def name(self) -> str:
        return f"old({self.model.params.tag}"

    @property
    def dataset(self) -> Dataset:
        return self.model.dataset

    def can_classify(self, trace: SyscallTrace) -> bool:
        return len(trace) > self.window_size + 1

    def hash_data(self) -> dict:
        return {
            **super().hash_data(),
            "model": self.model.params.hashcode,
            "tag": self.model.params.tag,
            "sequence_length": self.window_size,
            "old": True,
        }

    @property
    def feature_size(self) -> int:
        return self.model.feature_size

    def likelihood(self, sequence: np.ndarray) -> float:
        """Returns the average negative log likelihood of a sequence."""
        total_likelihood = 0.0

        if len(sequence) < 2:
            return 0.0

        for subsequence, actual in self.subsequences(sequence):
            prediction: tf.Tensor = self.model(subsequence)[0]
            step_likelihood = prediction[actual].numpy()
            if step_likelihood > 0:
                total_likelihood -= math.log(step_likelihood)
            else:
                return sys.float_info.max / 100000

        gc.collect()
        tf.keras.backend.clear_session()

        return total_likelihood / (len(sequence) - 1)

    def subsequences(self, sequence: np.ndarray) -> List[Tuple[tf.Tensor, int]]:
        """
        Returns a list of tuples of all subsequences (converted to one-hot format) of the given sequence that start
        at index 0 and range from len(subsequence) == 1 to len(subsequence) == len(sequence) - 1, paired with the
        next element in the sequence.
        """
        def window_ending_at(idx, window_size):
            start_point = max(0, idx-window_size)
            subseq = sequence[start_point:idx]
            target = sequence[idx]

            return tf.convert_to_tensor(np.expand_dims(tf.one_hot(subseq, self.feature_size), 0)), target

        return [window_ending_at(i, self.window_size) for i in range(1, len(sequence))]
