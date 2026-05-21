import itertools
# noinspection PyUnresolvedReferences
import math
import os
from pathlib import Path
# noinspection PyUnresolvedReferences
from typing import List, Optional, Dict, Tuple, Union
# noinspection PyUnresolvedReferences
from collections import defaultdict

import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# noinspection PyUnresolvedReferences
import tensorflow as tf

# noinspection PyUnresolvedReferences
from simian_lstm.data import Dataset, SyscallTrace
# noinspection PyUnresolvedReferences
from simian_lstm.model import Model, HyperParameters
# noinspection PyUnresolvedReferences
from simian_lstm.classifier import CachedClassifier, EnsembleClassifier, \
    WindowingClassifier, TopKClassifier, ProbabilityMassClassifier
from simian_lstm.evaluation.roc import ROC
# noinspection PyUnresolvedReferences
from simian_lstm.evaluation.cluster import (ClusterSizeStats, ClusterCacheEntry, cluster_predicate,
                                            anti_cluster_predicate, cluster_cache)

plt.style.use(['default'])
plt.rcParams['figure.figsize'] = [8, 8]

# noinspection PyBroadException
try:
    windows_data = Dataset.load("windows")
except:
    pass

# noinspection PyBroadException
try:
    adfa = Dataset.load("adfa")
except:
    pass

base_path = Path("results")
figure_path = base_path / "figures"
cache_path = base_path / "cache.db"

STRATEGY = tf.distribute.MirroredStrategy()


def train(dataset: Dataset, tag: str, experiment: dict, model_version: int = 2, predicate=lambda _: True, distributed=False):
    """Trains a set of models, given experimental parameters from the experiment dict.
    Model version 2 will train an LSTM.
    Model version 3 will train a Transformer.
    Model version 1 is deprecated should not be used."""
    from simian_lstm.trial import train as inner_train
    if distributed:
        with STRATEGY.scope():
            inner_train(dataset, experiment, save_path=base_path, tag=tag,
                        model_version=model_version, predicate=predicate,
                        strategy=STRATEGY)
    else:
        inner_train(dataset, experiment, save_path=base_path, tag=tag,
                    model_version=model_version, predicate=predicate)


def model_check(
        models,
        output_name,
        ensemble=False,
        plot=True,
        best_only=False,
        common_subset=False,
        **kwargs
):
    """Worker function that does the bulk of the evaluation for evaluate()

    models: A list of Model objects

    output_name: specifies a human-friendly name for the graph to be saved with.

    ensemble: If true, the models will be ensembled using the technique
    described by Kim et al., and the ensemble will be evaulated.

    plot: If true, plot a graph of the ROC curves for the evaluated models

    best_only: If true, only plot and return the ROC for the best-performing model

    common_subset: If true, restrict evaluation to the common subset of traces
    compatible with all models. Otherwise, evaluate each model with all of its compatible traces.

    **kwargs: Forwarded to other functions deeper in the pipeline.
    """
    classifiers = [CachedClassifier.of_model(model, cache_path, **kwargs) for model in models]

    if ensemble:
        ensemble_class = EnsembleClassifier(classifiers.copy(), label="Ensemble")
        if "group_by" in kwargs:
            ensemble_class.calibrate_groups(**kwargs)
        else:
            ensemble_class.calibrate(**kwargs)
        classifiers.append(ensemble_class)

    if common_subset:
        kwargs["predicate"] = common_subset_predicate(classifiers, **kwargs)

    rocs = [ROC.of_classifier(c, **kwargs) for c in classifiers]

    if best_only:
        rocs = sorted(rocs, key=lambda r: r.auc)[-1:]

    if plot:
        ROC.plot_rocs(rocs, figure_path / f"{output_name}-roc.pdf", **kwargs)

    return rocs


def common_subset_predicate(classifiers, predicate=None, **kwargs):
    """Generates a predicate function for a collection of classifiers.
    The predicate takes a trace and returns True if the trace is compatible with all classifiers.

    classifiers: the list of classifiers

    predicate: if supplied, traces must also match this predicate to be included

    **kwargs: Forwarded to the Dataset.matching_traces function"""
    existing_predicate = predicate or (lambda t: True)

    def trace_id(trace: SyscallTrace):
        return trace.meta["category"], str(trace.meta["relpath"])

    common_traces = None

    for classifier in classifiers:
        def match_predicate(t):
            return classifier.can_classify(t) and existing_predicate(t)

        attack_traces = classifier.dataset.matching_traces("attack", predicate=match_predicate, **kwargs)
        test_traces = classifier.dataset.matching_traces("test", predicate=match_predicate, **kwargs)
        all_trace_ids = {trace_id(trace) for trace in itertools.chain(attack_traces, test_traces)}

        if common_traces is None:
            common_traces = all_trace_ids
        else:
            common_traces = all_trace_ids.intersection(common_traces)

    def _pred(trace: SyscallTrace):
        return existing_predicate(trace) and trace_id(trace) in common_traces

    return _pred


def sort_param(param):
    """Returns a function that extracts a given parameter from a model,
    for the purpose of sorting a list of models."""
    return lambda model: getattr(model.params, param)


def load_models(tags: Union[str, List[str]]) -> List[Model]:
    """Loads one or more models based on a tag or list of tags.
    Hashcodes are also accepted."""
    if isinstance(tags, str):
        tags = [tags]

    with STRATEGY.scope():
        return [model for tag in tags for model in Model.load_tag(base_path, tag, STRATEGY)]


def evaluate(
        tags: Union[str, List[str]],
        sort_by=None,
        output_name=None,
        **kwargs
) -> List[ROC]:
    """
    Evaluates a set of models, producing ROC curves.

    :param tags: A string or list of strings, which can be tags or hashcodes.
    :param sort_by: If specified, sort the models by some property. Can be either a function that extracts a sort key,
    or a string that will be converted to a function that extracts that key from the model's hyperparameters.
    :param output_name: A human-readable name for the output graph. If unspecified, will be generated by concatenating
    the tags of participating models.
    :param kwargs: Forwarded to functions deeper in the pipeline such as model_check.
    :return: A list of ROC objects.
    """
    models = load_models(tags)

    if sort_by:
        for sortfn in sort_by:
            if isinstance(sortfn, str):
                sortfn = sort_param(sortfn)
            models = sorted(models, key=sortfn)

    if not output_name:
        output_name = "-".join({model.tag for model in models})

    return model_check(models, output_name, **kwargs)


tag_check = evaluate


def plot_category_scores(class_scores: Dict[str, float], output_name, x_min=None, x_max=None, **_kwargs):
    """
    Plots a graph of scores of traces divided into various classes.
    :param class_scores: A dictionary mapping class names to scores.
    :param output_name: The name of the pdf file to save the graph as.
    If unspecified, graph will be displayed but not saved.
    :param x_min: The minimum value for the x-axis of the graph. If unspecified, defaults to min(scores)*0.975
    :param x_max: The maximum value for the x-axis of the graph. If unspecified, defaults to max(scores)*1.25
    :param _kwargs: Unused.
    """
    y_pos = np.arange(len(class_scores))
    labels = list(class_scores.keys())
    scores = [class_scores[cat] for cat in labels]

    fig, ax = plt.subplots()
    bars = ax.barh(y_pos, scores)
    bars[-1].set_color('salmon')
    if x_min is None:
        x_min = min(scores) * 0.975
    if x_max is None:
        x_max = max(scores) * 1.25
    plt.xlim(x_min, x_max)
    ax.set_yticks(y_pos, labels=labels)
    ax.bar_label(bars, fmt="%.3f")
    plt.xlabel("AUC")
    plt.ylabel("Category")

    if output_name:
        plt.savefig(figure_path / f"{output_name}.pdf")
    plt.show()


def auc_by_cluster(model_id, category="attack", output_name=None, n_calls=10, min_cluster_size=10, **kwargs):
    """
    Cluster the traces in a dataset category by frequency of the top N most used calls in the category, and graph the
     anomaly scores assigned to those clusters by a given model.
    :param model_id: The tag or hashcode of the model to use.
    If multiple models exist for the tag, pick one arbitrarily.
    :param category: One of "attack", "test", "training", or "validation"
    :param output_name: If specified, save the graph to {output_name}.pdf
    :param n_calls: the N in the top-N calls
    :param min_cluster_size: Minimum size of a cluster
    :param kwargs: Unused for now.
    :return: Cluster objects.
    """
    model = load_models(model_id)[0]
    dataset = model.dataset

    def make_label(cluster_id, cluster):
        if cluster_id == -1:
            # cluster_label = "Outliers"
            cluster_label = ""
        else:
            # cluster_label = f"Cluster {cluster_id}"
            cluster_label = f"C{cluster_id}"

        # return f"{cluster_label} ({len(cluster)}t)"
        return f"{len(cluster)}t"

    clusters = cluster_cache.lookup(dataset, category, n_calls, min_cluster_size).clusters
    rocs = {make_label(label, cluster): evaluate(model_id,
                                                 predicate=cluster_predicate(cluster),
                                                 plot=False,
                                                 quiet=True)[0]
            for label, cluster in clusters.items()}
    class_scores = {label: roc.auc for label, roc in rocs.items()}
    plot_category_scores(class_scores, output_name, **kwargs)
    return clusters
