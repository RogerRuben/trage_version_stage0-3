"""Evaluate Stage3 order models with calibration, slices, and order bootstrap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

STAGE2_SCRIPTS = Path(__file__).resolve().parents[2] / "stage2" / "scripts"
if str(STAGE2_SCRIPTS) not in sys.path: sys.path.insert(0, str(STAGE2_SCRIPTS))
from stage2_deep_v3_utils import ece_score, metric_dict  # noqa: E402


def parse_args():
    parser=argparse.ArgumentParser(); parser.add_argument("--rule-root",type=Path,required=True); parser.add_argument("--tabular-root",type=Path,required=True); parser.add_argument("--deepsets-root",type=Path,required=True); parser.add_argument("--warehouse-root",type=Path,required=True); parser.add_argument("--order-feature-root",type=Path,required=True); parser.add_argument("--output-root",type=Path,default=Path("stage3/output/evaluation")); parser.add_argument("--bootstrap-rounds",type=int,default=500); parser.add_argument("--seed",type=int,default=2026); return parser.parse_args()


def load_predictions(args):
    rule=pd.read_parquet(args.rule_root/"predictions.parquet"); rule["model"]="rule_q90"
    tab=pd.read_parquet(args.tabular_root/"predictions.parquet"); tab=tab[tab.feature_set.eq("rc_mstnet")].copy(); tab["model"]="order_lightgbm"
    deep=pd.read_parquet(args.deepsets_root/"predictions.parquet"); deep["model"]="deepsets_route_attention"
    return pd.concat([rule,tab,deep],ignore_index=True,sort=False)


def metadata(warehouse:Path,features:Path):
    parts=[]
    for split in ["validation","test"]:
        link=pd.read_parquet(next((warehouse/"link_predictions"/f"split={split}").glob("*.parquet")),columns=["order_id","estimated_link_entry_time","route_link_length_m","route_link_seq"])
        meta=link.groupby("order_id",as_index=False).agg(route_length_m=("route_link_length_m","sum"),link_count=("route_link_seq","size"),start_time=("estimated_link_entry_time","min")); meta["split"]=split
        feat=pd.read_parquet(features/f"split={split}"/"order_features.parquet",columns=["order_id","rc_lcs_uncertainty_q90","rc_pmis_uncertainty_q90","rc_rts_uncertainty_q90"]); meta=meta.merge(feat,on="order_id",validate="one_to_one"); parts.append(meta)
    meta=pd.concat(parts,ignore_index=True); local=pd.to_datetime(meta.start_time,utc=True).dt.tz_convert("Asia/Shanghai"); meta["peak_offpeak"]=np.where(local.dt.hour.isin([7,8,9,17,18,19]),"peak","offpeak"); meta["route_bucket"]=pd.cut(meta.link_count,[-1,20,60,np.inf],labels=["short","medium","long"]).astype(str); meta["uncertainty"] = meta[["rc_lcs_uncertainty_q90","rc_pmis_uncertainty_q90","rc_rts_uncertainty_q90"]].mean(axis=1); return meta


def main():
    args=parse_args(); args.output_root.mkdir(parents=True,exist_ok=True); predictions=load_predictions(args); meta=metadata(args.warehouse_root,args.order_feature_root); predictions=predictions.merge(meta,on=["order_id","split"],how="left",validate="many_to_one")
    metric_rows=[]; slice_rows=[]; calibrated=[]; rng=np.random.default_rng(args.seed); bootstrap=[]
    for (model,target), group in predictions.groupby(["model","target"]):
        required=["true_raw","true_tail","pred_raw","pred_probability"]
        val=group[group.split.eq("validation")].dropna(subset=required).copy(); test=group[group.split.eq("test")].dropna(subset=required).copy()
        logit=lambda p: np.log(np.clip(p,1e-5,1-1e-5)/(1-np.clip(p,1e-5,1-1e-5)))
        calibrator=LogisticRegression(C=1e6,max_iter=1000).fit(logit(val.pred_probability.to_numpy()).reshape(-1,1),val.true_tail.astype(int))
        test["pred_probability_calibrated"]=calibrator.predict_proba(logit(test.pred_probability.to_numpy()).reshape(-1,1))[:,1]
        residual=np.abs(val.true_raw-val.pred_raw); q=float(residual.quantile(.90)); test["lower"]=(test.pred_raw-q).clip(0,1); test["upper"]=(test.pred_raw+q).clip(0,1); test["covered"]=test.true_raw.between(test.lower,test.upper)
        metrics=metric_dict(test.true_raw.to_numpy(),test.pred_raw.to_numpy(),test.pred_probability_calibrated.to_numpy(),test.true_tail.to_numpy(bool)); metrics.update({"brier_calibrated":float(np.mean((test.pred_probability_calibrated-test.true_tail.astype(float))**2)),"ece_calibrated":ece_score(test.true_tail.to_numpy(float),test.pred_probability_calibrated.to_numpy()),"interval_90_coverage":float(test.covered.mean()),"interval_mean_width":float((test.upper-test.lower).mean())}); metric_rows.append({"model":model,"target":target,**metrics}); calibrated.append(test)
        for slice_name in ["route_bucket","peak_offpeak"]:
            for value,part in test.groupby(slice_name):
                m=metric_dict(part.true_raw.to_numpy(),part.pred_raw.to_numpy(),part.pred_probability_calibrated.to_numpy(),part.true_tail.to_numpy(bool)); slice_rows.append({"model":model,"target":target,"slice":slice_name,"value":value,**m})
        threshold=test.uncertainty.quantile(.8)
        for value,part in [("low80",test[test.uncertainty.lt(threshold)]),("high20",test[test.uncertainty.ge(threshold)])]:
            m=metric_dict(part.true_raw.to_numpy(),part.pred_raw.to_numpy(),part.pred_probability_calibrated.to_numpy(),part.true_tail.to_numpy(bool)); slice_rows.append({"model":model,"target":target,"slice":"uncertainty","value":value,**m})
        for replicate in range(args.bootstrap_rounds):
            idx=rng.integers(0,len(test),len(test)); part=test.iloc[idx]; m=metric_dict(part.true_raw.to_numpy(),part.pred_raw.to_numpy(),part.pred_probability_calibrated.to_numpy(),part.true_tail.to_numpy(bool)); bootstrap.append({"model":model,"target":target,"replicate":replicate,"auc":m["auc"],"ap":m["ap"],"lift_top10":m["lift_top10"]})
    metrics=pd.DataFrame(metric_rows); metrics.to_csv(args.output_root/"model_metrics.csv",index=False); pd.DataFrame(slice_rows).to_csv(args.output_root/"slice_metrics.csv",index=False); pd.concat(calibrated,ignore_index=True).to_parquet(args.output_root/"calibrated_test_predictions.parquet",index=False,compression="zstd")
    boot=pd.DataFrame(bootstrap); ci=[]
    for (model,target),group in boot.groupby(["model","target"]):
        for metric in ["auc","ap","lift_top10"]: ci.append({"model":model,"target":target,"metric":metric,"ci_low":group[metric].quantile(.025),"ci_high":group[metric].quantile(.975),"rounds":args.bootstrap_rounds})
    pd.DataFrame(ci).to_csv(args.output_root/"order_cluster_bootstrap_ci.csv",index=False)
    (args.output_root/"stage3_report.md").write_text("# Stage3 prototype evaluation\n\nThis is one strict temporal train/validation/test chain (17/18/19), not yet a three-fold Stage3 claim.\n\n"+metrics.to_markdown(index=False,floatfmt=".4f"),encoding="utf-8"); print(metrics.to_string(index=False))


if __name__=="__main__": main()
