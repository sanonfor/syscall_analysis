from typing import List

from .model import ModelClassifier
from ..data import SyscallTrace


class FilteringClassifier(ModelClassifier):
    @property
    def name(self) -> str:
        return f"filtered_{self.model.params.deny_calls}_calls({super().name})"

    @property
    def allow_calls(self) -> List[int]:
        return self.model.params.allow_calls

    @property
    def deny_calls(self) -> List[int]:
        return self.model.params.deny_calls

    def likelihood(self, trace: SyscallTrace) -> float:
        trace = trace.filter(self.allow_calls, self.deny_calls)
        return super().likelihood(trace)

    def hash_data(self) -> dict:
        return {
            **super().hash_data(),
            "filtering": True,
        }

    def can_classify(self, trace: SyscallTrace) -> bool:
        return super().can_classify(trace.filter(self.allow_calls, self.deny_calls))
