import argparse
import json
from pathlib import Path

from .data.dataset import Dataset

parser = argparse.ArgumentParser("lstm", description="LSTM Model")
subcommands = parser.add_subparsers(dest="command")

train_cmd = subcommands.add_parser("train")
train_cmd.add_argument("--dataset", help="The name of the dataset to load.", default="adfa")
train_cmd.add_argument("--params", help="A JSON file with experiment parameters to load.")
train_cmd.add_argument("--tag", help="A label to include with results from this experiment.")
train_cmd.add_argument("--model-version", type=int, default=3, help="The version of the language model to train")

args = parser.parse_args()
save_path = Path("results")


if args.command == "train":
    from .trial import train

    dataset = Dataset.load(args.dataset)
    if args.params:
        params_path = Path(args.params)
        test_params = json.loads(params_path.read_text())
    else:
        # Toy parameters for a quick check
        test_params = {
            "train_size_target": [10000],
            "val_size_target": [5000],
            "epochs": [3],
            "sequence_length": [20],
        }

    train(dataset, test_params, model_version=args.model_version, save_path=save_path, tag=args.tag)
else:
    parser.print_usage()
