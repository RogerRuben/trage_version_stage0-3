# 信号、来源与时间语义

## 现有数据，不是新标签生产

读取 `stage2/output_v4/causal_history_store/events/day=YYYYMMDD.parquet` 的 09–27。
19 个文件 SHA 与已有 history_store_manifest 完全一致，共 4,998,113 个唯一
`(date, history_order_id, traversal_id)` 事件，对应每天 10,000 个已筛选订单。
它只覆盖有直接测量的 traversal，不能代表全部路线 token 或原始候选订单。

| 信号 | 定义/用途 | 本次非空数/比例 | 必要限制 |
|---|---|---:|---|
| observed_sec_per_m | 直接观测时间/距离，s/m | 4,591,292 / 91.861% | 局部测量 pace，不是完整 link 行程时间 |
| crawl_time_share | 1<v<5 m/s 的直接 interval 时间占比 | 4,998,113 / 100% | 不包含 stop；不是拥堵概率 |
| stop_time_share | v<=1 m/s 的直接 interval 时间占比 | 4,998,113 / 100% | 无法单独区分红灯、排队、接客等原因 |
| speed_cv_bounded | CV/(CV+1) | 4,986,884 / 99.775% | **非空不等于 target-valid** |
| acceleration_rms_bounded | RMS/(RMS+1)，原尺度 m/s² | 2,306,632 / 46.150% | 需要有效相邻 interval pair；非空计数不是置信度 |
| lcs_available | 四组件综合严格可用标记 | 2,040,512 / 40.826% | 比单个组件可用更严格，不可互相替代 |

依据：`stage1/v3/primitives.py`、`stage1/config/stage1_label_schema_v3.json`。
原 source schema 声明 retrospective_post_trip_realized_label。

特别注意：`stage2/v4/stage1_adapter.py::_add_component_masks` 还要求
speed CV 至少两个 direct intervals、加速度有正 pair_count 与 weight。
这些计数在 history events 中没有保留。本批不将上述非空率写成有效监督率，
也不使用 CV/acceleration 生成硬状态等级，只保留描述值和非空支持量。

## 参考量不是同一对象

- 现有 RTS reference：09–24、条件 cohort 中位数，含 edge/time/weekday 等
  fallback；用于相对正常条件的 excess，不是 free-flow。
- RTS percentile：Train 条件相对位置，不是交通拥堵等级。
- 本批低 pace reference：方向 edge 的 Train order-day 平均 pace q20，
  新增分析产物，不覆盖 RTS、不送入模型；明确只是经验低 pace 代理。
- posted/inferred speed limit：速度域属性，不是当前通行速度，不作本批分母。

## 三种时间不能互换

1. `time_bin_30m`：原直接观测窗口**开始**对应的半小时。
2. `availability_timestamp`：原窗口**结束**；history builder 也令
   `feature_timestamp` 等于此值。它是建模上的最早可用时刻，不是实测上传延迟。
3. 决策/进入时间：订单 decision_time 与预计 link entry；不在当前 history 表中。

37,444 条事件开始桶与完成桶不同，占 0.749%。本批用完成时间桶，不偷用开始时
尚未完成的观测。351 条跨订单日期午夜的事件排除并逐日记录，占 0.007%。

对未来在线接口，必须满足 source availability < decision time，并排除自身订单；
不能仅凭 history 表中恒为 true 的 `self_order_excluded` 字段证明查询已做排除。
本批没有执行这种 query，也没有声称在线可部署。

## 冻结 M3 cache

盘点 Train `s3/cache/m3` 及 Validation `s3/cache/m3_validation` 共 19 天，
13,532,743 个预测 traversal；全部 cache SHA 与其现存 manifest 一致。
`decision_time_only` 和 `predicted_progression_only` 在这些 manifest 中均为 true。
checkpoint 与原冻结值绑定，见 `prediction_inventory.json`。

输出含 pred_crawl、pred_stop、pred_speed_cv、pred_acceleration_rms、
pred_pace_p50、travel_time_p50_s、allocated_distance_m、support_group_code。
pred_rts 虽留在 cache 中，按现有契约不进入运营兼容主线。

cache 以 `(date, order_id, traversal_id)` 为键；没有独立 direction-edge/entry-time
字段。未来做路段预测状态必须与冻结路线/特征 lineage 连接，并保留 forecast
horizon，不能把同一路段不同订单、不同时点的预测简单当作同期观测平均。

M3 fit 是 09–21；本批参考 fit 是 09–24，不能混称模型训练支持。
本批只核 schema 和来源，不运行 M3、不评估拥堵状态预测精度。

## 数据访问缺口

Stage1 原 bucket 列表和 traversal_labels 实际读取均返回 WinError 5。
本轮没有修改 ACL 或补生产。现存 hash-bound history 表允许上述描述分析继续，
但没有 direct_interval_count、观测窗口 start、持续时间、距离覆盖和 pair_count。
因此不能检验完整路段覆盖，也不能复核所有 Stage2 target masks。
这不是“字段为零”，也不是原始产品已丢失的证据。
