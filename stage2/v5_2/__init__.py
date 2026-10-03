"""Stage 2 v5.2 micro-condition transfer package.

The package is additive: frozen v5/v5.1 code and products remain untouched.
"""

from .contracts import RESEARCH_CONTRACT, Stage2V52ContractError

__all__ = ["RESEARCH_CONTRACT", "Stage2V52ContractError"]
