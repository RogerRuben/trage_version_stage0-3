# 候选交通状态定义与解释边界

本批的名字均带 PROXY：它们是直接观测支持的运行状态代理，不是人工真值、
标准服务水平等级、AV 安全能力或已授权运营限制。

## 拥堵程度轴

R = 窗口订单等权 pace / 该方向 edge 的 Train q20 pace。
L = 同一 joint-valid 订单集合上的平均 crawl_time_share + stop_time_share。

| 候选状态 | 条件（先满足支持规则） | 可解释为 |
|---|---|---|
| NON_CONGESTED_PROXY | R<1.5 且 L<0.5 | 相对参考未明显变慢，且低速时间不占多数 |
| CONGESTED_PROXY | R>=1.5 且 L>=0.5，未达下一行 | 相对变慢和低速占比共同支持 |
| SEVERE_CONGESTION_PROXY | R>=2 且 L>=0.8 | 更强的相对变慢和低速暴露共同支持 |
| MIXED_EVIDENCE | 相对 pace 和低速占比不一致 | 不强行解释成三类之一 |
| UNKNOWN | 参考/订单/跨度/新鲜度不足或缺值 | 未知，不是畅通 |

以上阈值是执行前写入配置的探索性选择。未根据 profile 区分度或派单结果优化，
也没有声称这些数值是交通工程统一标准。

FHWA 将 TTI 定义为相对 free-flow 的行程时间比，并有其特定 free-flow
估计方案。本批只有局部观测 pace 与经验 q20，因此刻意不用 TTI 名称，也不
把现有 RTS 的条件中位数换个名字当作 free-flow。
[FHWA 定义](https://ops.fhwa.dot.gov/perf_measurement/ucr/documentation.htm)。

## 不稳定程度轴

保留 speed_cv_bounded、acceleration_rms_bounded 的连续值和观测支持；本批
**不冻结平稳/不平稳的二元分界**。历史表缺少原始 pair/count 掩码，必须先
恢复这些测量证据才能将其升级为受支持的第二分类轴。

拥堵并不自动意味着高加速度波动；缓慢均匀排队和反复起停应可区别。
本批不把各组件加权成新风险指数，也不改变现有 E/Q/C。

## 仍未解决的识别问题

- q20 可能包含信号控制和长期低速条件，不一定近似真实自由流。
- 不同订单观测的是同一 link 的不同部分，partial-window pace 比值可能包含
  空间覆盖差别；更多订单不能自动消除这种偏差。
- stop 无原因标签，不能声称已区分拥堵、红灯或上下客。
- 低速阈值沿用既有 1/5 m/s，只是实现信号，不是按道路等级校准的拥堵标准。
- 19.22% 的非 UNKNOWN cell 仍是 MIXED_EVIDENCE，说明一维“三档”还不足。
- 本批没有人工 traffic truth，因此没有 accuracy、precision、recall 或安全率。

## 后续 profile 边界

第一批不分配 C/M/A 接受矩阵，不把 severe proxy 直接变成 C/M hard infeasible。
下一批先分析与 E/Q/C 的重叠，明确 UNKNOWN 与混合证据如何表达，随后才制定
显式假设性的 profile 暴露容忍度。不得为了让三类规模相近而调本批阈值。
