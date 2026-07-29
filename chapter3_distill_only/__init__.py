"""Register the Chapter 3 distillation-only Fairseq components."""

from __future__ import annotations

import sys

# AV-HuBERT treats a one-element argv as a request for its non-package debug
# imports.  Fairseq user-module imports and ``python -c`` legitimately have
# that shape, so add a temporary sentinel while registrations are imported.
_added_argv_sentinel = len(sys.argv) == 1
if _added_argv_sentinel:
    sys.argv.append("chapter3_distill_only_import")

try:
    from .criterion import (
        AVHubertDistillOnlyCriterion,
        AVHubertDistillOnlyCriterionConfig,
    )
    from .model import AVHubertDistillOnly, AVHubertDistillOnlyConfig
    from .student import Chapter3AVHubertConfig, Chapter3AVHubertModel
    from .task import AVHubertDistillOnlyTask, AVHubertDistillOnlyTaskConfig
finally:
    if _added_argv_sentinel:
        sys.argv.pop()

__all__ = [
    "AVHubertDistillOnly",
    "AVHubertDistillOnlyConfig",
    "AVHubertDistillOnlyCriterion",
    "AVHubertDistillOnlyCriterionConfig",
    "AVHubertDistillOnlyTask",
    "AVHubertDistillOnlyTaskConfig",
    "Chapter3AVHubertConfig",
    "Chapter3AVHubertModel",
]
