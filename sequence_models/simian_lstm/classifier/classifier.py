import json
from abc import ABCMeta, abstractmethod
from hashlib import sha256
from pathlib import Path
from typing import Optional, List

import numpy as np
from tqdm import tqdm

from simian_lstm.data import Dataset, SyscallTrace


class Classifier(metaclass=ABCMeta):
    def __init__(self):
        self._hashcode = None
        self.label: Optional[str] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """A human-readable name for the classifier."""
        pass

    @property
    @abstractmethod
    def dataset(self) -> Dataset:
        """The dataset the classifier's model was trained on."""
        pass

    @property
    def allow_calls(self) -> List[int]:
        """A list of system calls that the classifier is willing to examine. Traces with calls outside this list
        will be rejected."""
        return []

    @property
    def deny_calls(self) -> List[int]:
        """A list of system calls that the classifier refuses to examine.
        Traces including these calls will be rejected."""
        return []

    @abstractmethod
    def can_classify(self, trace: SyscallTrace) -> bool:
        return True

    @abstractmethod
    def likelihood(self, trace: SyscallTrace) -> float:
        pass

    def likelihood_multi(self, traces: List[SyscallTrace]) -> List[float]:
        return [self.likelihood(trace) for trace in traces]

    def group_likelihood(self, traces: List[SyscallTrace], aggregate: str = "max") -> float:
        scores = [self.likelihood(trace) for trace in traces]
        if aggregate == "max":
            return max(scores)
        elif aggregate == "weighted":
            weights = [len(trace) - 19 for trace in traces]
            weighted_scores = [score * weight for score, weight in zip(scores, weights)]
            return sum(weighted_scores) / sum(weights)
        else:
            raise ValueError(f"Unknown score aggregation type: {aggregate}")

    def seq_likelihood(self, seq: np.ndarray) -> float:
        trace = SyscallTrace.from_seq(seq)
        return self.likelihood(trace)

    @property
    def hashcode(self) -> str:
        if self._hashcode is None:
            self._hashcode = sha256(str(self.hash_data()).encode("utf-8")).hexdigest()
        return self._hashcode

    @property
    @abstractmethod
    def feature_size(self) -> int:
        pass

    @abstractmethod
    def hash_data(self) -> dict:
        return {}

    def score_dataset(
            self,
            dataset: Dataset,
            out_file: Path,
            min_len=25,
            max_len=0,
            requirements: Optional[dict] = None
    ):
        with out_file.open("a") as f:
            for category, traces in dataset.categories.items():
                print(f"Evaluating {category} traces.")
                traces = [trace.clip(max_len) for trace in traces if trace.match(min_len=min_len, **requirements)]

                for trace in tqdm(traces):
                    score = self.likelihood(trace)

                    result = {
                        "file": str(trace.file),
                        "score": score,
                        "category": category,
                    }

                    f.write(json.dumps(result) + "\n")
                    f.flush()
