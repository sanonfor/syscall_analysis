import json
from pathlib import Path


DIR = Path(__file__).parent


def attack_metadata(metadata):
    attack_dir = DIR / "attack"
    for file in attack_dir.glob("**/*.txt"):
        stem = file.parent.name.rsplit("_", maxsplit=1)[0]
        metadata[f"mal_{file.name}"] = {"attack_class": stem}


def benign_metadata(metadata):
    benign_dirs = [DIR / "training", DIR / "validation", DIR / "test"]
    for d in benign_dirs:
        for file in d.glob("**/*.txt"):
            metadata[f"ben_{file.name}"] = {"attack_class": None}


def main():
    metadata = {}
    benign_metadata(metadata)
    attack_metadata(metadata)
    metadata_file = DIR / "metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
