from collections import defaultdict
from typing import List

import numpy as np
from tqdm import tqdm

from simian_lstm.data import Dataset, SyscallTrace

from .classifier import Classifier


class EnsembleClassifier(Classifier):
    def __init__(self, classifiers: List[Classifier], label=None):
        super().__init__()
        self.label = label
        self.classifiers = classifiers
        self.biases = None
        self.group_biases = None

        # Since we don't know how important each classifier is, weight them equally
        weight = 1 / len(classifiers)
        self.weights = [weight for _ in classifiers]

    @property
    def name(self) -> str:
        return f"ensemble({','.join(c.name for c in self.classifiers)})"

    @property
    def dataset(self) -> Dataset:
        assert all(x.dataset == self.classifiers[0].dataset for x in self.classifiers)
        return self.classifiers[0].dataset

    @property
    def feature_size(self) -> int:
        return min(classifier.feature_size for classifier in self.classifiers)

    def hash_data(self) -> dict:
        return {
            **super().hash_data(),
            "classifiers": [classifier.hashcode for classifier in self.classifiers],
            "biases": self.biases,
            "weights": self.weights,
        }

    def calibrate(self, min_len=0, max_len=0, **_kwargs):
        """Calibrates the biases of each classifier by averaging their scores for benign training data."""
        training_data = [trace.clip(max_len) for trace in self.dataset.categories["training"]
                         if len(trace) >= min_len and self.can_classify(trace)]
        assert training_data

        scores = defaultdict(list)
        for trace in tqdm(training_data):
            for classifier in self.classifiers:
                scores[classifier.hashcode].append(classifier.likelihood(trace))

        self.biases = {c_hashcode: np.median(c_scores) for c_hashcode, c_scores in scores.items()}

    def calibrate_groups(self, group_by="report", min_len=0, max_len=0, **_kwargs):
        training_groups = self.dataset.grouped_traces("training", group_by=group_by, min_len=min_len, max_len=max_len)
        assert training_groups

        scores = defaultdict(list)
        for group in tqdm(training_groups):
            for classifier in self.classifiers:
                scores[classifier.hashcode].append(classifier.group_likelihood(group))

        self.group_biases = {c_hashcode: np.median(c_scores) for c_hashcode, c_scores in scores.items()}

    def combine_scores(self, scores: List[float], biases) -> float:
        def leaky_relu(x: float) -> float:
            return max(x, 0.001 * x)

        total = 0.0

        for classifier, weight, score in zip(self.classifiers, self.weights, scores):
            bias = biases[classifier.hashcode]
            total += weight * leaky_relu(score - bias)

        return total

    def likelihood(self, trace: SyscallTrace) -> float:
        assert self.biases is not None, "Need to calibrate before classifying"
        scores = [classifier.likelihood(trace) for classifier in self.classifiers]
        return self.combine_scores(scores, self.biases)

    def group_likelihood(self, traces: List[SyscallTrace], aggregate: str = "max") -> float:
        assert self.group_biases is not None, "Need to calibrate before classifying"
        scores = [classifier.group_likelihood(traces, aggregate=aggregate) for classifier in self.classifiers]
        return self.combine_scores(scores, self.group_biases)

    def can_classify(self, trace: SyscallTrace) -> bool:
        return all(c.can_classify(trace) for c in self.classifiers)
