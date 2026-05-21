import itertools
import random
from hashlib import sha256


def experiment_permutations(param_dict):
    keys, values = zip(*param_dict.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def hashrand(key: str) -> random.Random:
    sha = sha256()
    sha.update(key.encode('utf-8'))
    return random.Random(sha.hexdigest())


def np_uniq(a):
    import numpy as np
    # Acts like unix `uniq` utility, compacting repeated sequences of the same value in a numpy array
    return a[np.r_[True, a[:-1] != a[1:]]]


def np_filter(trace):
    # Returns a new trace based on the input trace, with repeated calls compacted by np_uniq
    from simian_lstm.data.syscall_trace import SyscallTrace
    uniq_calls = np_uniq(trace.trace)
    filtered = SyscallTrace(trace.file, trace.meta, trace.max_len)
    filtered._trace = uniq_calls
    return filtered
