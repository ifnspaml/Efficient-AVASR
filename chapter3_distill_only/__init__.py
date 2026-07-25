"""Chapter 3 distillation-only Fairseq user module.

Importing this package registers the isolated task, native student model,
distillation wrapper, and criterion.  No checkpoint or dataset is touched at
import time.
"""

from __future__ import annotations

import sys

# AV-HuBERT's legacy debug-import switch treats a one-element argv as a script
# invocation and attempts non-package imports.  Unit tests and ``python -c``
# legitimately have that shape, so make package imports deterministic without
# permanently changing process arguments.
try:
    import torch as _torch  # noqa: F401
    from fairseq.models import register_model as _register_model  # noqa: F401
except (ImportError, ModuleNotFoundError):
    # Manifest, selection, and reporting utilities intentionally work in the
    # lightweight login environment, which has no PyTorch runtime.  A Fairseq
    # training process always satisfies these imports and follows the
    # registration branch below.
    RUNTIME_AVAILABLE = False
    __all__ = []
else:
    RUNTIME_AVAILABLE = True
    _added_argv_sentinel = len(sys.argv) == 1
    if _added_argv_sentinel:
        sys.argv.append("chapter3_distill_only_import")

    try:
        from .criterion import (  # noqa: F401
            AVHubertDistillOnlyCriterion,
            AVHubertDistillOnlyCriterionConfig,
        )
        from .model import (  # noqa: F401
            AVHubertDistillOnly,
            AVHubertDistillOnlyConfig,
        )
        from .student import (  # noqa: F401
            Chapter3AVHubertConfig,
            Chapter3AVHubertModel,
        )
        from .task import (  # noqa: F401
            AVHubertDistillOnlyTask,
            AVHubertDistillOnlyTaskConfig,
        )
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
