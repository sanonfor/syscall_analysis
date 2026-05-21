from __future__ import annotations

import math
from hashlib import sha256
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import tensorflow as tf
# noinspection PyProtectedMember
from numpy.lib.stride_tricks import sliding_window_view


class SyscallTrace:
    def __init__(self, file: Path, meta: Dict[str, Any], max_len=0):
        self.file = file
        self.meta = meta
        self._trace = None
        self.max_len = max_len
        self._hash = None

    def data(
            self,
            sequence_length: int,
            deduplicate: bool = False,
    ) -> Optional[np.ndarray]:
        if len(self.trace) < sequence_length:
            return None

        windows = sliding_window_view(self.trace, sequence_length)

        if deduplicate:
            windows = np.unique(windows, axis=0)

        return windows

    def clip(self, max_len: int) -> SyscallTrace:
        if max_len:
            clipped = SyscallTrace(self.file, self.meta, max_len)
            clipped._trace = self.trace[:max_len]
            return clipped
        else:
            return self

    def filter(self, allow: List[int], deny: List[int]) -> SyscallTrace:
        """
        Filter a trace by removing all calls from the deny list (if present)
        and all calls not present on the allow list (if present).
        """
        filtered_trace = self.trace

        if allow:
            filtered_trace = filtered_trace[np.in1d(filtered_trace, allow)]

        if deny:
            filtered_trace = filtered_trace[np.in1d(filtered_trace, deny, invert=True)]

        filtered = SyscallTrace(self.file, self.meta, self.max_len)
        filtered._trace = filtered_trace
        return filtered

    def match(self, min_len=0, **kwargs):
        if len(self) <= min_len:
            return False

        for k, v in kwargs.items():
            if self.meta.get(k) != v:
                return False

        return True

    @property
    def trace(self):
        if self._trace is None:
            self._trace = np.fromstring(self.file.read_text(), dtype=np.int32, sep=" ")
            if self.max_len:
                self._trace = self.trace[:self.max_len]

        return self._trace

    def __len__(self):
        return len(self.trace)

    def one_hot(self, feature_size: int) -> np.ndarray:
        return tf.one_hot(self.trace, feature_size).numpy()

    @property
    def tensor(self):
        return tf.convert_to_tensor(self.trace)

    @property
    def syscalls(self):
        return set(self.trace)

    def hexdigest(self) -> str:
        if self._hash is None:
            self._hash = sha256(self.trace.data.tobytes()).hexdigest()
        return self._hash

    @staticmethod
    def pick_windows(
            traces: List[SyscallTrace],
            sequence_length: int,
            max_sequences: int = math.inf,
            output_sequences: bool = False,
            deduplicate: bool = False,
            deduplicate_traces: bool = False,
            **_kwargs,
    ):
        sequences = []
        total = 0
        for trace in traces:
            s = trace.data(sequence_length=sequence_length+1, deduplicate=deduplicate_traces)
            if s is None:
                continue
            sequences.append(s)
            total += len(s)

            if total > max_sequences:
                break

        sequences = np.concatenate(sequences, axis=0)

        if deduplicate:
            sequences = np.unique(sequences, axis=0)

        if output_sequences:
            targets = sequences[:, 1:]
        else:
            targets = sequences[:, -1:]

        sequences = sequences[:, :-1]
        return sequences, targets

    @classmethod
    def from_seq(cls, seq: np.ndarray) -> SyscallTrace:
        seq_hash = sha256(str(seq).encode("utf-8")).hexdigest()
        trace = SyscallTrace(Path(f"/tmp/{seq_hash}"), {}, 0)
        trace._trace = seq
        return trace

    @classmethod
    def load_dir(cls, path: Path, metadata: Dict[str, Dict[str, Any]]):
        malicious = "attack" in path.name
        prefix = "mal_" if malicious else "ben_"

        def load_trace(f: Path):
            relpath = f.relative_to(path)
            meta = {
                "relpath": relpath,
                "category": path.name,
                "group": relpath.parent.name or relpath.name,
                "malicious": malicious,
                **metadata.get(f"{prefix}{f.name}", {})
            }
            return SyscallTrace(f, meta=meta)

        return [load_trace(f) for f in path.glob("**/*.txt")]
