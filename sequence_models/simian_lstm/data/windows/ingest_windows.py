import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

from simian_lstm.data.encode import canonize_encoding
from simian_lstm.data.ingest import partition_data, ingest_files, pid_groups

parser = argparse.ArgumentParser(
    "ingest_windows.py",
    description="Ingests the windows dataset and converts it into the format used by this repository."
)

parser.add_argument("in_path", help="The directory containing the dataset.")
parser.add_argument("out_path", help="The directory to write the output to.")


def read_metadata(data_path: Path, prefix: str = "") -> Dict[str, Dict[str, Any]]:
    lookup = defaultdict(lambda: defaultdict(lambda: {}))
    metadata_path = data_path / "metadata_perProcess.csv"

    with metadata_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            report = row["reportLabel"]
            pid = row["pid"]
            lookup[report][pid]["tid_main"] = str(row["tid_main"])
            lookup[report][pid]["main_process"] = row["isMainProcess"] == "True"

    metadata = {}
    for file in data_path.glob("**/*.txt"):
        try:
            report, pid, tid = file.stem.rsplit("_", maxsplit=2)
            metadata[f"{prefix}{file.name}"] = {
                "main_thread": lookup[report][pid].get("tid_main", None) == tid,
                "main_process": lookup[report][pid].get("main_process", False),
                "report": report,
                "pid": pid,
            }
        except Exception as e:
            print(file.name)
            print(e)
            sys.exit(1)

    return metadata


def ingest(traces_path: Path, out_path: Path, encoding: Dict[str, int]):
    benign_path = traces_path / "ben"
    attack_path = traces_path / "mal"

    ben_metadata = read_metadata(benign_path, prefix="ben_")
    mal_metadata = read_metadata(attack_path, prefix="mal_")
    metadata = ben_metadata.copy()
    metadata.update(mal_metadata)
    metadata_path = out_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    benign_files = list(benign_path.glob("**/*.txt"))
    groups = partition_data(benign_files, pid_groups)
    groups["attack"] = list(attack_path.glob("**/*.txt"))

    for group_name, group_files in groups.items():
        group_path = out_path / group_name
        group_path.mkdir(exist_ok=True)
        ingest_files(group_files, group_path, encoding)


def main():
    args = parser.parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.mkdir(exist_ok=True)

    encoding_path = Path(__file__).parent / "encoding_canon.json"
    encoding = json.loads(encoding_path.read_text())

    ingest(in_path, out_path, encoding)

    encoding_out_path = out_path / "encoding.json"
    encoding_out_path.write_text(json.dumps(encoding, indent=2))
    feature_size = max(list(encoding.values())) + 1
    feature_size_path = out_path / "feature_size.txt"
    feature_size_path.write_text(str(feature_size))

    canon_encoding_path = out_path / "encoding_canon.json"
    canon_encoding_path.write_text(json.dumps(canonize_encoding(encoding), indent=2))


if __name__ == "__main__":
    main()
