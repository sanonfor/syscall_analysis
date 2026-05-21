from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from sklearn.cluster import HDBSCAN
import numpy as np

from simian_lstm.data import Dataset, SyscallTrace


MIN_TRACE_LEN = 21


@dataclass
class ClusterCacheEntry:
    dataset: Dataset
    category: str
    n_calls: int
    min_cluster_size: int
    clusters: Dict[int, List[SyscallTrace]]
    centroids: Dict[int, list]
    medoids: Dict[int, list]


class ClusterCache:
    def __init__(self):
        self.cache = {}

    def lookup(self, dataset: Dataset, category: str, n_calls: int, min_cluster_size: int) -> ClusterCacheEntry:
        cache_label = (dataset.name, category, n_calls, min_cluster_size)
        if cache_label not in self.cache:
            traces = [trace for trace in dataset.categories[category] if len(trace) > MIN_TRACE_LEN]
            clusters, centroids, medoids = cluster_traces(traces, n_calls=n_calls, min_cluster_size=min_cluster_size)
            sorted_keys = sorted(list(clusters.keys()), reverse=True)
            clusters = {k: clusters[k] for k in sorted_keys}
            self.cache[cache_label] = ClusterCacheEntry(dataset, category, n_calls,
                                                        min_cluster_size, clusters, centroids, medoids)

        return self.cache[cache_label]


cluster_cache = ClusterCache()


@dataclass
class ClusterSizeStats:
    cluster_scores: Dict[int, List[float]]
    cluster_aucs: Dict[int, float]
    clusters: ClusterCacheEntry

    @property
    def total_traces(self):
        return sum([len(cluster) for cluster in self.clusters.clusters.values()])

    def below(self, threshold=0.2):
        below_traces = [len(self.clusters.clusters[idx]) for idx, auc in self.cluster_aucs.items() if auc < threshold]
        return sum(below_traces) / self.total_traces
        # return len([auc for auc in self.cluster_aucs.values() if auc < threshold]) / len(self.cluster_aucs)

    def above(self, threshold=0.8):
        above_traces = [len(self.clusters.clusters[idx]) for idx, auc in self.cluster_aucs.items() if auc > threshold]
        return sum(above_traces) / self.total_traces
        # return len([auc for auc in self.cluster_aucs.values() if auc > threshold]) / len(self.cluster_aucs)


def top_n_calls(traces, n):
    call_counts = defaultdict(lambda: 0)
    for trace in traces:
        for call in trace.trace:
            call_counts[call] += 1
    sorted_counts = sorted(call_counts.items(), key=lambda x: x[1], reverse=True)
    top_n = [k for k, v in sorted_counts[:n]]
    return top_n


# Convert traces
def trace_to_vec(trace, top_n):
    call_counts = defaultdict(lambda: 0)
    for call in trace.trace:
        call_counts[call] += 1
    return [call_counts[c]/len(trace) for c in top_n]


def cluster_traces(traces, n_calls=10, min_cluster_size=20):
    top_n = top_n_calls(traces, n_calls)
    attack_vectors = np.array([trace_to_vec(trace, top_n) for trace in traces], dtype=np.double)
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, store_centers="both")
    hdb.fit(attack_vectors)
    clusters = defaultdict(list)
    for label, trace in zip(hdb.labels_, traces):
        clusters[label].append(trace)

    centroids = defaultdict(list)
    medoids = defaultdict(list)
    for label in clusters.keys():
        if label < 0:
            continue
        centroids[label] = hdb.centroids_[label]
        medoids[label] = hdb.medoids_[label]

    return clusters, centroids, medoids


def cluster_predicate(cluster):
    malicious = cluster[0].meta["malicious"]

    def _p(trace):
        return (trace.meta["malicious"] != malicious) or (trace in cluster)

    return _p


def anti_cluster_predicate(cluster):
    malicious = cluster[0].meta("malicious")

    def _p(trace):
        return (trace.meta["malicious"] != malicious) or (trace not in cluster)

    return _p
