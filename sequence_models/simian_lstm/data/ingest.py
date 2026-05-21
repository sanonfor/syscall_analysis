import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable

from simian_lstm.data.encode import add_encoding

FileGroups = Tuple[Dict[str, List[Path]], Dict[str, int]]
GroupingFn = Callable[[List[Path]], FileGroups]


def pid_groups(files: List[Path], min_tokens: int = 20) -> FileGroups:
    groups = defaultdict(list)
    counts = defaultdict(lambda: 0)

    for file in files:
        pid = file.parent.name
        count = len(file.read_text().split())
        if count < min_tokens:
            continue

        groups[pid].append(file)
        counts[pid] += count

    return groups, counts


def no_groups(files: List[Path]) -> FileGroups:
    groups = {}
    counts = {}

    for file in files:
        group_name = str(file)
        count = len(file.read_text().split())

        groups[group_name] = [file]
        counts[group_name] = count

    return groups, counts


def partition_data(files: List[Path], grouping_fn: GroupingFn, training_pct=0.6, val_pct=0.2) -> Dict[str, List[Path]]:
    groups, counts = grouping_fn(files)
    pids = list(groups.keys())
    total_calls = sum(counts.values())

    rand = random.Random(1)
    rand.shuffle(pids)

    training_count = int(total_calls * training_pct)
    val_count = int(total_calls * val_pct)

    def take_pids(start_idx: int, call_count: int) -> int:
        i = start_idx
        total = 0
        while i < len(pids) and total < call_count:
            total += counts[pids[i]]
            i += 1

        return i

    def collect_traces(pids: List[str]) -> List[Path]:
        traces = []
        for pid in pids:
            traces.extend(groups[pid])
        return traces

    val_start = take_pids(0, training_count)
    test_start = take_pids(val_start, val_count)

    training_pids = pids[:val_start]
    val_pids = pids[val_start:test_start]
    test_pids = pids[test_start:]

    training_set = set(training_pids)
    val_set = set(val_pids)
    test_set = set(test_pids)
    assert len(training_pids) == len(training_set)
    assert len(val_pids) == len(val_set)
    assert len(test_pids) == len(test_set)
    assert not training_set.intersection(val_set)
    assert not training_set.intersection(test_set)
    assert not test_set.intersection(val_set)

    report = {
        "training_total": sum(counts[pid] for pid in training_pids),
        "val_total": sum(counts[pid] for pid in val_pids),
        "test_total": sum(counts[pid] for pid in test_pids)
    }
    print(json.dumps(report, indent=2))
    print(f"total_calls: {total_calls}")
    print(f"training_count: {training_count}")
    print(f"val_count: {val_count}")
    print(f"val_start: {val_start}\ntest_start: {test_start}")
    print(f"#pids: {len(pids)}")

    partitions = {
        "training": collect_traces(training_pids),
        "validation": collect_traces(val_pids),
        "test": collect_traces(test_pids)
    }

    return partitions


def ingest_file(file: Path, out_path: Path, encoding: Dict[str, int]):
    in_sequence = file.read_text().split()
    out_sequence = []

    for c in in_sequence:
        c = c.strip()
        try:
            out_sequence.append(str(encoding[c]))
        except KeyError:
            add_encoding(encoding, c)

    out_path.write_text(" ".join(out_sequence))


def ingest_files(files: List[Path], out_dir: Path, encoding: Optional[Dict[str, int]]):
    for file in files:
        out_path = out_dir / file.name
        if encoding:
            ingest_file(file, out_path, encoding)
        else:
            shutil.copyfile(file, out_path)


def calculate_feature_size(dirs: List[Path]) -> int:
    max_feature = 0

    for d in dirs:
        for file in d.glob("**/*.txt"):
            max_call = max(int(call) for call in file.read_text().split())
            max_feature = max(max_feature, max_call)

    return max_feature + 1