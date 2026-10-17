"""Small static route-length input. No fare, realized time, or future demand."""
from math import isfinite
from pathlib import Path


class EconomicDistanceStore:
    def __init__(self, frame):
        expected={"date","order_id","service_distance_m"}
        frame=frame.copy()
        for key in ("date","order_id"):
            if key in frame: frame[key]=frame[key].astype(str)
        if set(frame.columns)!=expected or frame.duplicated(["date","order_id"]).any():
            raise ValueError("economic route lengths require one row per date/order and a geometry-only schema")
        self.values={}
        for row in frame.itertuples(index=False):
            distance=float(row.service_distance_m)
            if not isfinite(distance) or distance<=0:
                raise ValueError("missing/invalid full route distance; no OD-distance fallback")
            self.values[str(row.date),str(row.order_id)]=distance

    @classmethod
    def load(cls, path):
        import pandas as pd
        import hashlib,json
        path=Path(path)
        manifest=json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        digest=hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda:stream.read(1<<20),b""): digest.update(block)
        if (manifest.get("status")!="ECONOMIC_STATIC_INPUTS_PREPARED_NO_SIMULATION"
                or digest.hexdigest()!=manifest.get("output_sha256")):
            raise ValueError("economic route-length preparation manifest does not match its product")
        result=cls(pd.read_parquet(path)); result.source_sha256=digest.hexdigest()
        return result

    def distance(self, date, order_id):
        try:
            return self.values[str(date),str(order_id)]
        except KeyError as error:
            raise ValueError("economic input is missing this declared route identity") from error

    def attach_history(self, frame):
        result=frame.copy()
        if any(str(d)>"20161024" for d in result.date):
            raise ValueError("future target-day route lengths cannot enter forecast history")
        result["service_distance_m"]=[self.distance(d,o) for d,o in result[["date","order_id"]].itertuples(index=False,name=None)]
        return result


def aggregate_distances(path, date, wanted):
    """One projected Arrow batch at a time; never accumulate route-edge frames."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    source=pq.ParquetFile(path)
    columns=["order_id","route_part_length_m"]
    wanted=set(map(str,wanted)); totals={}; route_rows=0
    value_set=pa.array(sorted(wanted),type=source.schema_arrow.field("order_id").type)
    for batch in source.iter_batches(batch_size=65536,columns=columns,use_threads=False):
        table=pa.Table.from_batches([batch])
        table=table.filter(pc.is_in(table["order_id"],value_set=value_set))
        if not len(table): continue
        lengths=table["route_part_length_m"].to_numpy()
        if not np.isfinite(lengths).all() or (lengths<0).any():
            raise ValueError("invalid frozen route-part lengths")
        route_rows+=len(table)
        sums=table.group_by("order_id",use_threads=False).aggregate([("route_part_length_m","sum")])
        for oid,distance in zip(sums["order_id"].to_pylist(),sums["route_part_length_m_sum"].to_pylist()):
            oid=str(oid); totals[oid]=totals.get(oid,0.)+float(distance)
    if set(totals)!=wanted or any(not isfinite(v) or v<=0 for v in totals.values()):
        raise ValueError("incomplete economic route-length population")
    return [dict(date=str(date),order_id=oid,service_distance_m=totals[oid]) for oid in sorted(totals)],route_rows
