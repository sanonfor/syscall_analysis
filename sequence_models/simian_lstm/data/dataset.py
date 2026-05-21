import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Hashable
from weakref import WeakValueDictionary

import tensorflow as tf

from simian_lstm.data.syscall_trace import SyscallTrace
from simian_lstm.data.training_data import TrainingData
from simian_lstm.util import hashrand


DATASET_CACHE = WeakValueDictionary()
TracePredicate = Callable[[SyscallTrace], bool]


class Dataset:
    categories: Dict[str, List[SyscallTrace]]
    metadata: Dict[str, Dict[str, Any]]

    def __init__(self, path: Path):
        self.path = path
        self.name = path.name

        meta_path = path / "metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
        else:
            metadata = {}

        self.categories = {
            p.name: SyscallTrace.load_dir(p, metadata)
            for p in path.iterdir() if p.is_dir()
        }
        self.feature_size = int((path / "feature_size.txt").read_text().strip())

        encoding_path = path / "encoding_canon.json"
        if encoding_path.exists():
            self.encoding = json.loads(encoding_path.read_text())
            self.decoding = {v: k for k, v in self.encoding.items()}
        else:
            self.encoding = {}
            self.decoding = {}

    # Convenience methods
    @property
    def attack(self):
        return self.categories["attack"]

    @property
    def training(self):
        return self.categories["training"]

    @property
    def validation(self):
        return self.categories["validation"]

    @property
    def test(self):
        return self.categories["test"]

    @property
    def traces(self):
        from itertools import chain
        return list(chain(*self.categories.values()))

    def data_windows(
            self,
            category: str,
            sequence_length: int,
            max_windows: int,
            max_len=0,
            hashkey: Optional[str] = None,
            shuffle: bool = True,
            batch_size=128,
            one_hot: bool = False,
            **kwargs,
    ) -> tf.data.Dataset:
        traces = self.matching_traces(category, max_len=max_len, **kwargs)
        if shuffle:
            rand = hashrand(hashkey or self.name)
            rand.shuffle(traces)

        def label_data(sequence, target):
            if one_hot:
                sequence = tf.one_hot(sequence, self.feature_size)
            return (
                {
                    "input_sequence": sequence,
                },
                target,
            )

        data = tf.data.Dataset.from_tensor_slices(
            SyscallTrace.pick_windows(traces, sequence_length, max_windows, **kwargs)
        ).batch(batch_size).map(label_data, num_parallel_calls=tf.data.AUTOTUNE)

        return data

    def training_data(
            self,
            sequence_length: int,
            max_training: int,
            max_validation: int,
            max_len=0,
            hashkey: Optional[str] = None,
            **kwargs,
    ) -> TrainingData:
        training = self.data_windows(
            "training",
            sequence_length,
            max_training,
            max_len=max_len,
            hashkey=hashkey,
            **kwargs
        )
        validation = self.data_windows(
            "validation",
            sequence_length,
            max_validation,
            max_len=max_len,
            hashkey=hashkey,
            **kwargs
        )

        training_metadata = {k: v for k, v in kwargs.items() if not callable(v)}

        return TrainingData(
            training,
            validation,
            feature_size=self.feature_size,
            dataset=self.name,
            meta={**training_metadata, "max_len": max_len},
        )

    def matching_traces(
            self,
            category: str,
            min_len=0,
            max_len=0,
            requirements=None,
            allow_calls: Optional[List[int]] = None,
            deny_calls: Optional[List[int]] = None,
            predicate: Optional[TracePredicate] = lambda _: True,
            trace_filter: Optional[Callable[[SyscallTrace], SyscallTrace]] = lambda x: x,
            **_kwargs,
    ):
        if requirements is None:
            requirements = {}

        traces = self.categories[category]
        if allow_calls or deny_calls:
            traces = [trace.filter(allow_calls, deny_calls) for trace in traces]

        traces = [trace_filter(trace) for trace in traces]

        return [trace.clip(max_len)
                for trace in traces
                if trace.match(min_len=min_len, **requirements) and predicate(trace)]

    @classmethod
    def load(cls, dataset_name: str):
        global DATASET_CACHE

        if dataset_name not in DATASET_CACHE:
            path = Path(__file__).parent / dataset_name
            dataset = Dataset(path)
            DATASET_CACHE[dataset_name] = dataset

        return DATASET_CACHE[dataset_name]

    def grouped_traces(
            self,
            category: str,
            group_by: str = "report",  # "report" or "pid"
            group_fn: Callable[[SyscallTrace], Hashable] = None,
            **kwargs
    ) -> List[List[SyscallTrace]]:
        if group_fn is None:
            if group_by == "pid":
                def _group_fn(t: SyscallTrace) -> str:
                    return f"{t.meta['report']}_{t.meta['pid']}"
            else:
                def _group_fn(t: SyscallTrace) -> str:
                    return t.meta[group_by]
            group_fn = _group_fn

        groups = defaultdict(list)
        for trace in self.matching_traces(category, **kwargs):
            groups[group_fn(trace)].append(trace)

        return list(groups.values())
