import argparse
from csv import DictReader
from pathlib import Path
from typing import Dict

import numpy as np

from simian_lstm.data import SyscallTrace


def load_table(path: Path) -> Dict[int, str]:
    result: Dict[int,str] = {}
    with path.open() as f:
        reader = DictReader(f)
        for row in reader:
            result[int(row['x86'])] = row['x86_name'].strip()

    return result


default_table_path = Path(__file__).absolute().parent / "adfa-translation.csv"

parser = argparse.ArgumentParser("translate.py")
parser.add_argument("--table", type=Path, default=default_table_path)
parser.add_argument("file", type=Path)


def translate_trace(table: Dict[int, str], trace: SyscallTrace):
    return [table.get(call, str(call)) for call in trace.trace]


def main():
    args = parser.parse_args()
    table = load_table(args.table)
    trace = SyscallTrace(args.file, {})

    for call in translate_trace(table, trace):
        print(call)


if __name__ == "__main__":
    main()
