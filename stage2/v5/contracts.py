"""Contracts and validation errors for the isolated Stage 2 v5 implementation."""

from __future__ import annotations


class Stage2V5ContractError(ValueError):
    """Raised when a frozen v5 data, temporal, or model contract is violated."""

