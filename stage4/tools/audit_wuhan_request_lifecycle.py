"""Descriptive audit of the public completed-trip sample; no simulator or model fitting."""
import hashlib
import json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / 'output/external_research/wuhan_robotaxi_2026/repository/7-sample_data_and_code/carpooling_one_day_sample_code/v12-traj_sample_2024-11-12.csv'
OUT = BASE / 'docs/wuhan_robotaxi_review/request_lifecycle'
FIELDS = ['呼单时间','接单时间','出发接驾时间','到达起点时间','开始行程时间','到达目的地时间','取消时间']
PAIRS = {
    'request_to_accept': ('呼单时间','接单时间'),
    'accept_to_depart': ('接单时间','出发接驾时间'),
    'depart_to_arrival': ('出发接驾时间','到达起点时间'),
    'arrival_to_boarding_proxy': ('到达起点时间','开始行程时间'),
    'boarding_proxy_to_destination': ('开始行程时间','到达目的地时间'),
    'request_to_arrival': ('呼单时间','到达起点时间'),
    'request_to_boarding_proxy': ('呼单时间','开始行程时间'),
    'request_to_destination': ('呼单时间','到达目的地时间'),
    'accept_to_arrival': ('接单时间','到达起点时间'),
    'accept_to_destination': ('接单时间','到达目的地时间'),
}

def describe(s):
    v = s.dropna()
    return {'n':int(v.size),'missing':int(s.isna().sum()),
            **{k:float(v.quantile(p)) if len(v) else None for k,p in [('min',0),('p25',.25),('p50',.5),('p75',.75),('p90',.9),('p99',.99),('max',1)]},
            'negative':int((v<0).sum()),'zero':int((v==0).sum()),
            'gt_300s':int((v>300).sum()),'gt_1800s':int((v>1800).sum())}

def main():
    # Only 3 MB / 6,727 rows. No IDs, addresses or per-order results are exported.
    d = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    t = {f:pd.to_datetime(d[f],format='%Y/%m/%d %H:%M',errors='coerce') for f in FIELDS}
    intervals = {k:(t[b]-t[a]).dt.total_seconds() for k,(a,b) in PAIRS.items()}
    result = {'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'rows':len(d),'unique_orders':d['订单号'].nunique(),'unique_vehicles':d['车辆id'].nunique(),
              'method':'Descriptive only; pandas linear quantiles; no cleaning/exclusion/model fitting.',
              'fields':{},'intervals':{k:describe(v) for k,v in intervals.items()}}
    for f in FIELDS:
        result['fields'][f] = {'blank':int(d[f].eq('').sum()),'parse_failure_nonblank':int((t[f].isna() & d[f].ne('')).sum()),
                               'minute_format':int(d[f].str.fullmatch(r'\d{4}/\d{1,2}/\d{1,2} \d{1,2}:\d{2}').sum()),
                               'date_counts':t[f].dt.strftime('%Y-%m-%d').value_counts(dropna=False).to_dict()}
    result['status_counts'] = d['订单状态'].value_counts().to_dict()
    result['processing_markers'] = {f: d[f].value_counts().to_dict() for f in ['取消次数','虚拟单合并数']}
    result['cancel_field'] = {
        'nonblank_completed':int((d['取消时间'].ne('') & d['订单状态'].eq('完成')).sum()),
        'equals_destination':int((t['取消时间'].eq(t['到达目的地时间']) & t['取消时间'].notna()).sum()),
        'earlier_than_request':int((t['取消时间']<t['呼单时间']).sum())}
    adjacent = list(PAIRS)[:5]
    violations = pd.concat([intervals[k]<0 for k in adjacent],axis=1).any(axis=1)
    result['chronology'] = {'any_adjacent_negative':int(violations.sum()),'complete_chain':int(pd.DataFrame({k:t[k] for k in FIELDS[:-1]}).notna().all(axis=1).sum())}
    result['duration_comparisons'] = {}
    for field in ['接驾时长','行程时长']:
        v = pd.to_numeric(d[field],errors='coerce')
        comparisons = {}
        for name, interval in intervals.items():
            error = interval-v
            comparisons[name] = {'available_pairs':int(error.notna().sum()),
                                 'abs_error_le_60s':int(error.abs().le(60).sum()),
                                 'median_abs_error_s':float(error.abs().median()),
                                 'p90_abs_error_s':float(error.abs().quantile(.9))}
        result['duration_comparisons'][field] = {'reported_numeric':describe(v),'comparisons':comparisons,
                                                'integer_share':float((v.dropna()%1==0).mean())}
    # Descriptive groups; neither hypothesis tests nor city-transfer estimates.
    grouped = []
    merge_count = pd.to_numeric(d['虚拟单合并数'],errors='coerce')
    merge_group = merge_count.map(lambda x: 'marker_eq_1' if x == 1 else ('marker_gt_1' if x > 1 else 'marker_other'))
    for group_field, groups in [('request_hour', t['呼单时间'].dt.hour), ('order_channel',d['订单来源']), ('merge_marker',merge_group)]:
        for group in sorted(groups.dropna().unique()):
            mask = groups.eq(group)
            for metric in ['request_to_arrival','request_to_boarding_proxy','arrival_to_boarding_proxy']:
                grouped.append({'group_field':group_field,'group':str(group),'metric':metric,**describe(intervals[metric][mask])})
    result['vehicle_order_count_distribution'] = describe(d.groupby('车辆id').size())
    result['vehicle_order_count_distribution'] = {k:v for k,v in result['vehicle_order_count_distribution'].items() if k not in ['gt_300s','gt_1800s']}
    # Magnitude screens only, not exclusion thresholds.
    result['tail_screen'] = {'arrival_to_boarding_gt_30min':int((intervals['arrival_to_boarding_proxy']>1800).sum()),
                             'boarding_before_arrival':int((intervals['arrival_to_boarding_proxy']<0).sum()),
                             'pickup_wait_gt_30min':int((intervals['request_to_arrival']>1800).sum())}
    result['zero_duration_crosscheck'] = {
        'reported_trip_duration_zero_but_boarding_to_destination_positive':int((pd.to_numeric(d['行程时长'],errors='coerce').eq(0) & intervals['boarding_proxy_to_destination'].gt(0)).sum()),
        'reported_pickup_duration_missing_but_depart_to_arrival_present':int((pd.to_numeric(d['接驾时长'],errors='coerce').isna() & intervals['depart_to_arrival'].notna()).sum())}
    total = sum(intervals[k] for k in adjacent)
    checks = {'rows_match_prior_inventory':len(d)==6727,
              'unique_orders':d['订单号'].nunique()==len(d),
              'interval_decomposition_exact':bool((total==intervals['request_to_destination']).all()),
              'arrival_quantiles_match_previous':bool(intervals['request_to_arrival'].quantile(.5)==300 and intervals['request_to_arrival'].quantile(.9)==720),
              'boarding_median_matches_previous':bool(intervals['request_to_boarding_proxy'].quantile(.5)==360)}
    if not all(checks.values()): raise AssertionError(checks)
    result['calculation_checks'] = checks
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(grouped).to_csv(OUT/'grouped_timing.csv',index=False)
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__ == '__main__':
    main()
