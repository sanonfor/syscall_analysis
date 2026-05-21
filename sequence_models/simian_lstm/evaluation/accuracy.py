from pathlib import Path
from typing import List

from matplotlib import pyplot as plt

from simian_lstm.model.model import Model


def plot_training(models: List[Model], save_path: Path, title=None, labelfn=lambda m: m.label, metric="accuracy"):
    fig, ax = plt.subplots()

    for model in models:
        history = model.tags["history"]
        training_accuracy = history[metric]
        val_accuracy = history[f"val_{metric}"]
        ticks = range(1, len(training_accuracy)+1)

        label = labelfn(model)

        ax.plot(ticks, training_accuracy, label=f"{label} Training")
        ax.plot(ticks, val_accuracy, label=f"{label} Validation")

    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.title())
    ax.set_title(title or f"Training vs Validation {metric.title()}")
    plt.savefig(save_path)
