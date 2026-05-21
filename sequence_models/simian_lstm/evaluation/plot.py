import json
from pathlib import Path

from matplotlib import pyplot as plt

from simian_lstm.model import Model


def plot_scores(base_path: Path, model: Model, graph_path):
    score_path = model.scores_path(base_path)
    trace_scores = list(map(json.loads, score_path.read_text().splitlines()))

    from collections import defaultdict
    categories = defaultdict(lambda: [])

    for score in trace_scores:
        categories[score["category"]].append(score["score"])

    fig, ax = plt.subplots()
    ax.set_title(f"Scores for {model.params.tag} on dataset '{model.stats['dataset']['dataset']}'")
    ax.boxplot(categories.values(), labels=categories.keys())
    plt.savefig(graph_path)
