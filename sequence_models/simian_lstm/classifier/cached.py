from __future__ import annotations

from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Optional, List
import sqlite3

from .topk import TopKClassifier
from .model import ModelClassifier
from .old import OldClassifier
from .windowing import WindowingClassifier
from .classifier import Classifier
from simian_lstm.data import SyscallTrace, Dataset
from ..model import Model


class ClassifierCache(metaclass=ABCMeta):
    """Abstract base class for classifier caches. Caches map (classifier, trace) -> score so that traces that have
    already been scored do not need to be repeatedly recomputed."""
    @abstractmethod
    def get_by_digest(self, classifier: str, trace: str) -> Optional[float]:
        """Retrieves the score using hexadecimal digests of the classifier and trace."""
        pass

    def get(self, classifier: Classifier, trace: SyscallTrace) -> Optional[float]:
        return self.get_by_digest(classifier.hashcode, trace.hexdigest())

    @abstractmethod
    def put_by_digest(self, classifier: str, trace: str, score: float):
        pass

    def put(self, classifier: Classifier, trace: SyscallTrace, score: float):
        self.put_by_digest(classifier.hashcode, trace.hexdigest(), score)

    @staticmethod
    def from_path(path) -> ClassifierCache:
        path = Path(path)
        if path.suffix == ".db":
            return SQLCache(path)


class SQLCache(ClassifierCache):
    """An implementation of the classifier cache backed by sqlite3."""
    def __init__(self, path: Path):
        self.path = path
        self.connection = sqlite3.connect(self.path)
        self.cursor = self.connection.cursor()
        self.init()

    def init(self):
        self.cursor.executescript('''
        CREATE TABLE IF NOT EXISTS cache
        (
          classifier TEXT,
          trace TEXT,
          score REAL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS cache_idx ON cache (classifier, trace);
        ''')
        self.connection.commit()

    def get_by_digest(self, classifier: str, trace: str) -> Optional[float]:
        res = self.cursor.execute(
            "SELECT score FROM cache WHERE classifier = ? AND trace = ?;",
            (classifier, trace)
        ).fetchone()

        if res:
            return res[0]
        else:
            return None

    def put_by_digest(self, classifier: str, trace: str, score: float):
        self.cursor.execute(
            "INSERT INTO cache (classifier, trace, score) VALUES (?, ?, ?);",
            (classifier, trace, score)
        )
        self.connection.commit()


class CachedClassifier(Classifier):
    """A classifier that is backed by a cache, falling back to a base classifier on cache misses."""
    def __init__(self, base_classifier: Classifier, cache_path: Path):
        super().__init__()
        self.base_classifier = base_classifier
        self.label = base_classifier.label
        self.cache_path = cache_path
        self.cache = ClassifierCache.from_path(cache_path)

    @property
    def name(self) -> str:
        return self.base_classifier.name

    @property
    def dataset(self) -> Dataset:
        return self.base_classifier.dataset

    @property
    def allow_calls(self) -> List[int]:
        return self.base_classifier.allow_calls

    @property
    def deny_calls(self) -> List[int]:
        return self.base_classifier.deny_calls

    @property
    def feature_size(self) -> int:
        return self.base_classifier.feature_size

    def can_classify(self, trace: SyscallTrace) -> bool:
        return self.base_classifier.can_classify(trace)

    def likelihood(self, trace: SyscallTrace) -> float:
        cached_score = self.cache.get(self.base_classifier, trace)
        if cached_score is not None:
            return cached_score

        score = self.base_classifier.likelihood(trace)
        self.cache.put(self.base_classifier, trace, score)
        return score

    def hash_data(self) -> dict:
        return self.base_classifier.hash_data()

    @classmethod
    def of_model(
            cls,
            model: Model,
            cache_path: Path,
            sequence_length=None,
            batch_size=None,
            window_size=None,
            old=False,
            top_k=0,
            **kwargs,
    ):
        """Utility method for constructing cached classifiers."""
        if old:
            base_classifier = OldClassifier(model, window_size=sequence_length)
        elif window_size:
            base_classifier = WindowingClassifier(model, sequence_length=sequence_length,
                                                  batch_size=batch_size, window_size=window_size, **kwargs)
        elif top_k:
            base_classifier = TopKClassifier(model, k=top_k,
                                             sequence_length=sequence_length, batch_size=batch_size, **kwargs)
        else:
            base_classifier = ModelClassifier(model, sequence_length=sequence_length, batch_size=batch_size)
        return cls(base_classifier, cache_path)
