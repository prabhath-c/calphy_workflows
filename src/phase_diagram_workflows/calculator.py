import sys

from .calphy import calculator as _calculator

sys.modules[__name__] = _calculator
