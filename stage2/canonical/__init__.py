"""Canonical Stage 2 dispatch-time prediction primitives."""

from .dispatch_time import (
    DISPATCH_FEATURE_WHITELIST,
    FORBIDDEN_DISPATCH_FIELDS,
    attach_dispatch_snapshot,
    audit_dispatch_features,
    hierarchical_fallback,
)

__all__ = [
    "DISPATCH_FEATURE_WHITELIST",
    "FORBIDDEN_DISPATCH_FIELDS",
    "attach_dispatch_snapshot",
    "audit_dispatch_features",
    "hierarchical_fallback",
]
