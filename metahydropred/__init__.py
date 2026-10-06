"""MetaHydroPred prediction core: input validation -> baseline models -> meta-model -> back-transform."""
import os
import sys

# backtransform.py (repository root) is the single definition of y = exp(z) - 1, shared with the training scripts.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
