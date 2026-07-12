"""Low-complexity rolling stacking on common LightGBM/RC-MSTNet rows."""

from __future__ import annotations

import argparse,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression,Ridge

SCRIPT_DIR=Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0,str(SCRIPT_DIR))
from stage2_deep_v3_utils import LINK_TARGETS,metric_dict  # noqa:E402
from summarize_stage2_deep_v3_fold_metrics import aligned_fold  # noqa:E402

def parse_args():
 p=argparse.ArgumentParser();p.add_argument("--deep-prediction-root",type=Path,required=True);p.add_argument("--lightgbm-oof",type=Path,required=True);p.add_argument("--output-root",type=Path,default=Path("stage2/output/deep_v3/stacking"));return p.parse_args()

def main():
 args=parse_args();args.output_root.mkdir(parents=True,exist_ok=True);lgbm=pd.read_parquet(args.lightgbm_oof);rows=[];predictions=[]
 for train_fold,test_fold in [(1,2),(2,3)]:
  for target in LINK_TARGETS:
   train=aligned_fold(args.deep_prediction_root/f"fold={train_fold}"/"test_predictions.parquet",lgbm,train_fold,target,"static_rolling_dynamic_topology_route");test=aligned_fold(args.deep_prediction_root/f"fold={test_fold}"/"test_predictions.parquet",lgbm,test_fold,target,"static_rolling_dynamic_topology_route")
   raw_features=["deep_raw","lgbm_raw"];prob_features=["deep_prob","lgbm_prob"]
   reg=Ridge(alpha=1.0).fit(train[raw_features],train.true_raw);clf=LogisticRegression(C=1.0,max_iter=1000).fit(train[prob_features],train.true_tail.astype(int));stack_raw=np.clip(reg.predict(test[raw_features]),0,1);stack_prob=clf.predict_proba(test[prob_features])[:,1]
   for model,raw,prob in [("LightGBM",test.lgbm_raw.to_numpy(),test.lgbm_prob.to_numpy()),("RC-MSTNet",test.deep_raw.to_numpy(),test.deep_prob.to_numpy()),("shallow_stacking",stack_raw,stack_prob)]:
    rows.append({"train_fold":train_fold,"test_fold":test_fold,"target":target.upper(),"model":model,"common_rows":len(test),"common_orders":test.order_id.nunique(),**metric_dict(test.true_raw.to_numpy(),raw,prob,test.true_tail.to_numpy(bool))})
   predictions.append(pd.DataFrame({"order_id":test.order_id,"date":test.date,"route_link_id":test.planned_link_id,"route_link_seq":test.planned_link_seq,"target":target,"train_fold":train_fold,"test_fold":test_fold,"true_raw":test.true_raw,"true_tail":test.true_tail,"stack_raw":stack_raw,"stack_probability":stack_prob}))
 metrics=pd.DataFrame(rows);metrics.to_csv(args.output_root/"stacking_metrics.csv",index=False);pd.concat(predictions,ignore_index=True).to_parquet(args.output_root/"stacking_predictions.parquet",index=False,compression="zstd");(args.output_root/"stacking_report.md").write_text("# Stage2 shallow stacking\n\nCoverage is limited to common historical LightGBM OOF rows.\n\n"+metrics.to_markdown(index=False,floatfmt=".4f"),encoding="utf-8");print(metrics.to_string(index=False))
if __name__=="__main__":main()
