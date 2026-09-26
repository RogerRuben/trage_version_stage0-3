"""Same-unit shadow diagnostics for AV candidate opportunities.

Every count uses an ``(order, available AV, decision epoch)`` opportunity as
its unit. The helpers do not mutate requests, vehicles, candidate graphs, or
solver inputs. Sparse Top-K and routing sub-gates are retained explicitly so
that their loss is never mislabeled as a patience loss.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

import pandas as pd


GATE_COLUMNS = (
    "gate_av_n0_spatial",
    "gate_av_n1_passenger_compatible",
    "gate_av_n2_structurally_ready",
    "gate_av_n3_evidence_complete",
    "gate_av_n3a_shared_topk",
    "gate_av_n3b_route_returned",
    "gate_av_n4_pickup_within_patience",
    "gate_av_n5_solver_eligible",
    "gate_av_n6_selected",
)

LOSS_COLUMNS = (
    "gate_av_loss_acceptance",
    "gate_av_loss_no_selected_route",
    "gate_av_loss_hard_infeasible",
    "gate_av_loss_hard_unknown",
    "gate_av_loss_evidence_incomplete",
    "gate_av_loss_shared_topk",
    "gate_av_loss_routing_failure",
    "gate_av_loss_patience",
    "gate_av_loss_other_arc_condition",
    "gate_av_loss_dispatch_competition",
)


def empty_gate_counts() -> dict[str, int]:
    return {name: 0 for name in (*GATE_COLUMNS, *LOSS_COLUMNS)}


def selected_route_established(request: Any) -> bool:
    value = str(getattr(request, "selected_route_type", "")).strip().upper()
    return value not in {"", "NONE", "NAN", "NULL"}


def structural_reason(request: Any) -> str | None:
    """Return the mutually exclusive failed structural gate, or ``None``."""
    if not selected_route_established(request):
        return "NO_SELECTED_ROUTE"
    hard_state = str(getattr(request, "hard_state", "UNKNOWN")).strip().upper()
    if hard_state == "INFEASIBLE":
        return "HARD_INFEASIBLE"
    if hard_state != "FEASIBLE":
        return "HARD_UNKNOWN"
    return None


def evidence_contract_complete(request: Any, exposure: Any | None) -> bool:
    """Mirror the frozen exposure + AV service-time evidence contract."""
    service_time = float(getattr(request, "predicted_service_time_s", float("nan")))
    return (
        bool(getattr(request, "evidence_complete", False))
        and exposure is not None
        and isfinite(service_time)
        and service_time > 0.0
    )


def validate_gate_counts(row: dict[str, Any]) -> None:
    values = [int(row[name]) for name in GATE_COLUMNS]
    if any(value < 0 for value in values):
        raise ValueError("AV gate counts must be non-negative")
    if any(left < right for left, right in zip(values, values[1:])):
        raise ValueError(f"AV gate counts are not nested: {values}")
    identities = {
        "acceptance": (
            row["gate_av_n0_spatial"] - row["gate_av_n1_passenger_compatible"],
            row["gate_av_loss_acceptance"],
        ),
        "structural": (
            row["gate_av_n1_passenger_compatible"]
            - row["gate_av_n2_structurally_ready"],
            row["gate_av_loss_no_selected_route"]
            + row["gate_av_loss_hard_infeasible"]
            + row["gate_av_loss_hard_unknown"],
        ),
        "evidence": (
            row["gate_av_n2_structurally_ready"]
            - row["gate_av_n3_evidence_complete"],
            row["gate_av_loss_evidence_incomplete"],
        ),
        "topk": (
            row["gate_av_n3_evidence_complete"] - row["gate_av_n3a_shared_topk"],
            row["gate_av_loss_shared_topk"],
        ),
        "routing": (
            row["gate_av_n3a_shared_topk"] - row["gate_av_n3b_route_returned"],
            row["gate_av_loss_routing_failure"],
        ),
        "patience": (
            row["gate_av_n3b_route_returned"]
            - row["gate_av_n4_pickup_within_patience"],
            row["gate_av_loss_patience"],
        ),
        "other": (
            row["gate_av_n4_pickup_within_patience"]
            - row["gate_av_n5_solver_eligible"],
            row["gate_av_loss_other_arc_condition"],
        ),
        "competition": (
            row["gate_av_n5_solver_eligible"] - row["gate_av_n6_selected"],
            row["gate_av_loss_dispatch_competition"],
        ),
    }
    mismatches = {
        name: (int(expected), int(observed))
        for name, (expected, observed) in identities.items()
        if int(expected) != int(observed)
    }
    if mismatches:
        raise ValueError(f"AV gate loss identities failed: {mismatches}")


def aggregate_gate_epochs(epoch: pd.DataFrame, bin_minutes: int = 15) -> pd.DataFrame:
    required = {"timestamp", *GATE_COLUMNS, *LOSS_COLUMNS}
    missing = required.difference(epoch.columns)
    if missing:
        raise ValueError(f"gate epoch log missing columns: {sorted(missing)}")
    frame = epoch[["timestamp", *GATE_COLUMNS, *LOSS_COLUMNS]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    if frame["timestamp"].dt.tz is None:
        frame["timestamp"] = frame["timestamp"].dt.tz_localize("Asia/Shanghai")
    else:
        frame["timestamp"] = frame["timestamp"].dt.tz_convert("Asia/Shanghai")
    frame["time_bin_start"] = frame["timestamp"].dt.floor(f"{int(bin_minutes)}min")
    result = (
        frame.groupby("time_bin_start", as_index=False, sort=True)[
            [*GATE_COLUMNS, *LOSS_COLUMNS]
        ]
        .sum()
        .reset_index(drop=True)
    )
    for row in result.to_dict("records"):
        validate_gate_counts(row)
    return result
