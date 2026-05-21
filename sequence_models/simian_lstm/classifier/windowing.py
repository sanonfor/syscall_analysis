import numpy as np
# noinspection PyProtectedMember
from numpy.lib.stride_tricks import sliding_window_view

from simian_lstm.classifier.model import DEFAULT_BATCH_SIZE, ModelClassifier
from simian_lstm.data import SyscallTrace
from simian_lstm.model import Model


class WindowingClassifier(ModelClassifier):
    """
    Variant of ModelClassifier that scores traces by their worst sliding window of
    window_size calls, rather than their overall score.
    """
    def __init__(
            self,
            model: Model,
            sequence_length=None,
            batch_size=None,
            window_size=40,
            label=None,
            geometric=False,
            decay=0.5,
            additive_base=1,
            **_kwargs,
    ):
        super().__init__(model, sequence_length=sequence_length, batch_size=batch_size, label=label)
        self.window_size = window_size
        self.geometric = geometric
        self.decay = decay
        self.additive_base = additive_base

    @property
    def name(self) -> str:
        return f"worst_{self.window_size}({self.model.params.tag})"

    def hash_data(self) -> dict:
        return {
            **super().hash_data(),
            "window_size": self.window_size,
            "geometric": self.geometric,
            "decay": self.decay,
            "additive_base": self.additive_base,
        }

    def geomean(self, a):
        # Input shape: (batch_size, window_size)
        x = np.array(a)
        if self.decay:
            decay_coefficients = np.exp(np.arange(-self.window_size+1, 1, 1) * self.decay)
            x = x * decay_coefficients

        x = x + self.additive_base
        return np.exp(np.log(x).mean(axis=1))

    def likelihood(self, trace: SyscallTrace, batch_size=DEFAULT_BATCH_SIZE) -> float:
        if len(trace) < 2:
            return 0.0

        steps = self.step_likelihoods(trace)
        step_windows = sliding_window_view(steps, self.window_size)
        if self.geometric:
            window_means = self.geomean(step_windows)
        else:
            window_means = np.mean(step_windows, axis=1)

        return window_means.max(initial=0.0)

    def can_classify(self, trace: SyscallTrace) -> bool:
        return super().can_classify(trace) and len(trace) > (self.window_size + self.sequence_length + 5)
