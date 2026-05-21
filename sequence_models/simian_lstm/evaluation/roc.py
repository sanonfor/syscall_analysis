from __future__ import annotations

from pathlib import Path
from typing import List

from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm

from simian_lstm.classifier.classifier import Classifier


class ROC:
    def __init__(self, classifier: Classifier, auc: float, fpr: List[float], tpr: List[float], thresholds: List[float]):
        self.classifier = classifier
        self.auc = auc
        self.fpr = fpr
        self.tpr = tpr
        self.thresholds = thresholds

    @classmethod
    def of_classifier(
            cls,
            classifier: Classifier,
            quiet=False,
            predicate=None,
            aggregate="max",
            **kwargs
    ) -> ROC:
        if predicate:
            def match_predicate(t):
                return classifier.can_classify(t) and predicate(t)
        else:
            match_predicate = classifier.can_classify
        match_criteria = {
            "allow_calls": kwargs.get("allow_calls", classifier.allow_calls),
            "deny_calls": kwargs.get("deny_calls", classifier.deny_calls),
            "predicate": match_predicate,
        }

        if "group_by" in kwargs:
            negative_groups = classifier.dataset.grouped_traces("test", **match_criteria, **kwargs)
            positive_groups = classifier.dataset.grouped_traces("attack", **match_criteria, **kwargs)
            negative_scores = [classifier.group_likelihood(group, aggregate=aggregate) for group in
                               tqdm(negative_groups)]
            positive_scores = [classifier.group_likelihood(group, aggregate=aggregate) for group in
                               tqdm(positive_groups)]
        else:
            negatives = classifier.dataset.matching_traces("test", **match_criteria, **kwargs)
            positives = classifier.dataset.matching_traces("attack", **match_criteria, **kwargs)
            negative_scores = [classifier.likelihood(trace) for trace in tqdm(negatives, disable=quiet)]
            positive_scores = [classifier.likelihood(trace) for trace in tqdm(positives, disable=quiet)]

        return cls.of_scores(classifier, negative_scores, positive_scores)

    @classmethod
    def of_scores(cls, classifier: Classifier, negatives: List[float], positives: List[float]) -> ROC:
        all_scores = negatives + positives
        labels = [0] * len(negatives) + [1] * len(positives)
        auc = roc_auc_score(y_true=labels, y_score=all_scores)
        curve = roc_curve(y_true=labels, y_score=all_scores)

        return ROC(classifier, auc, *curve)

    def points(self):
        return list(zip(self.fpr, self.tpr))

    def stats(self):
        return {
            "auc": self.auc,
            "fpr": list(self.fpr),
            "tpr": list(self.tpr),
            "thresholds": list(self.thresholds),
            "classifier": self.classifier.name,
            "classifier_data": self.classifier.hash_data(),
        }

    @property
    def label(self) -> str:
        return f"{self.classifier.label or self.classifier.name}"

    def plot(self, ax, label: str):
        ax.plot(self.fpr, self.tpr, label=label)

    @classmethod
    def plot_rocs(cls, rocs: List[ROC], save_path: Path, labelfn=None, labels=None, title=None, **_kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        ax.plot([0, 1], [0, 1], color="lightgrey", linestyle="--")

        if labels is None:
            def apply_label(r: ROC):
                if r.classifier.label:
                    return r.classifier.label
                elif labelfn:
                    return labelfn(r)
                else:
                    return r.label

            labels = [apply_label(r) for r in rocs]

        for roc, label in zip(rocs, labels):
            roc.plot(ax, f"{label} (AUC: {roc.auc:.3f})")

        ax.legend(loc="lower right")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title or "Receiver Operating Characteristic")
        ax.set_aspect("equal", adjustable="box")
        plt.savefig(save_path)
