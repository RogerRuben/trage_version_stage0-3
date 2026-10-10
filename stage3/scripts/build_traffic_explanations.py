"""Build small Validation-only sidecars from existing batch2, no inference."""
import json
import time
from pathlib import Path

import pandas as pd
import psutil

from stage3.odd_tod.traffic_explanation import build_explanations, attach_explanations, SOURCE_COLUMNS
from stage3.scripts.traffic_state_batch1 import sha, write_json, write_parquet


def main():
    start = time.monotonic()
    root = Path("stage3/output/traffic_explanation_v1")
    root.mkdir(parents=True, exist_ok=True)
    profile = Path("stage3/config/stage3_av_capability_profiles.json")
    before = sha(profile)
    summary = {"status": "EXPLANATION_ONLY_COMPLETE", "dates": [],
               "profile_sha256": before, "training": False, "dispatch": False,
               "test31_read": False, "gpu_used": False, "decision_logic_modified": False}
    peak = 0
    for date in ("20161025", "20161026", "20161027"):
        source = Path(f"stage3/output/traffic_state_batch2/route_overlap_date={date}.parquet")
        source_sha = sha(source)
        # Daily narrow reads; no cross-day token accumulation or CDF loading.
        day = pd.read_parquet(source, columns=["date", "order_id", *SOURCE_COLUMNS,
                                               "rho_dynamic_C", "rho_dynamic_M", "rho_dynamic_A"])
        sidecar = build_explanations(day)
        audit = day[["date", "order_id", "rho_dynamic_C", "rho_dynamic_M", "rho_dynamic_A"]].copy()
        audit["date"] = audit.date.astype(str)
        audit["order_id"] = audit.order_id.astype(str)
        audit["selected_route_reference"] = "ORIGINAL:" + audit.order_id
        enriched = attach_explanations(audit, sidecar)
        pd.testing.assert_frame_equal(enriched[audit.columns], audit)
        output = root / f"date={date}.parquet"
        write_parquet(output, sidecar)
        assert sha(source) == source_sha
        summary["dates"].append({"date": date, "rows": len(sidecar), "source": str(source),
            "source_sha256": source_sha, "output": str(output), "output_sha256": sha(output),
            "unchanged_dynamic_ratio_rows": len(audit),
            "unknown_exposure_routes": int(sidecar.traffic_unknown_share.gt(0).sum()),
            "mixed_exposure_routes": int(sidecar.traffic_mixed_evidence_share.gt(0).sum())})
        peak = max(peak, psutil.Process().memory_info().rss / 2**20)
    assert sha(profile) == before
    summary.update(runtime_s=time.monotonic()-start, sampled_peak_rss_mb=peak,
                   rows=sum(d["rows"] for d in summary["dates"]))
    write_json(Path("stage3/docs/traffic_explanation_v1/summary.json"), summary)
    write_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
