# 第一批：描述性交通状态（执行前规则）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: authorized descriptive analysis
- Origin Date: 2026-09-27
- Verification Status: UNVERIFIED before execution
- Version Label: traffic_state_batch1.1

## 边界

只做信号盘点、支持度、候选状态定义、描述性验证。用户已授权第一批。
不训练、不推断、不派单、不修改 profile、旧标签或论文；不读 Test31。
数据来源为已存在且逐日 SHA 可与原 manifest 对账的 history events。
Stage1 原 bucket 当前不可访问；不能声称本轮重新验证了其逐 interval 产品。

## 时间与估计对象

- 方向身份：observed_directed_edge_uid，正反方向绝不混合。
- 分桶：Asia/Shanghai 30 分钟；按 availability_timestamp（观测窗口结束）
  归入 [t-1800,t)，输出 window_end=t。恰在 t 完成的事件归入下一桶。
- 原 time_bin_30m 来自观测窗口开始；单独报告它与完成时间桶的差别。
- 跨原订单日期午夜的少量事件显式排除并计数，防止同一实际时间桶分散
  在两份日表或 25 日完成的事件进入 09–24 参考池。
- 统计对象是完成于该窗口的局部直接观测，不是覆盖整个窗口/整条 link 的
  路况真值。不前向填充，不把没有观测的窗口解释为畅通。
- 同一订单在同一 edge/window 多次访问先等权平均，再跨订单等权。
  不重复加权长订单；订单不同不等于驾驶员独立。

## 参考和支持

- 09–24：每个方向 edge 的 order-day 平均有效 pace 的 q20，保留 q50 对照。
  这只是低 pace 经验参考，不称真实 free flow，不是原 RTS reference。
- 同时要求至少 100 个 order-days、5 个日期；不给无支持 edge 填道路等级均值。
- 当前窗口 joint_valid 要求有限正 pace、合法 crawl/stop 且两者和不超过 1。
- 当前窗口默认至少 3 个有效独立订单身份、完成时刻跨度至少 300 秒、最近证据
  距窗口末端不超过 600 秒。仅是探索性充分度门槛，不代表统计置信保证。
- 同时报告最低订单数 1/3/5 的覆盖变化，不根据结果重选默认值。
- 缺少直接 interval 数、观测持续时间、距离覆盖、原始起始时刻；相关字段应
  是不可得，而不是零。source lcs_available 与各组件非空分别报告。
- 09–24 结果为整段 Train 参考下的回顾性描述，不能作为这些天在线决策结果。
  25–27 使用更早的参考，但仍是完成事件的描述，不是前瞻预测验证。
- M3 实际训练日期 09–21，与描述参考 09–24 是不同对象；不将后者称模型支持。

## 候选状态规则（本批不赋予运营约束）

令 R=窗口订单等权 pace / Train edge q20；L=同一有效订单集合上的
mean(crawl+stop)。crawl 为 1<v<5 m/s，stop 为 v<=1 m/s（直接 interval 时间占比）。

1. 支持不足或缺值 → UNKNOWN。
2. R<1.5 且 L<0.5 → NON_CONGESTED_PROXY。
3. R>=2 且 L>=0.8 → SEVERE_CONGESTION_PROXY。
4. 其余 R>=1.5 且 L>=0.5 → CONGESTED_PROXY。
5. 其余 → MIXED_EVIDENCE，不能硬归类。

1.5/2 和 0.5/0.8 是本次明确提出的可解释候选划分，不是规范标准、学习出的
分类器或已获科学验证的阈值。stop 不单独触发严重拥堵；低速和相对变慢必须
同时出现。但这仍不能证明已排除红灯停车或接客停留。

运行不稳定轴保留 speed_cv_bounded、acceleration_rms_bounded 及各自支持量，
不在本批额外发明二元风险线。其是否适合命名为“停走状态”仍待验证。

参考：FHWA 的 TTI 以 free-flow 为分母，不能据此将本项目条件中位数或 q20
直接称标准 TTI；其 UCR 使用的 free-flow 定义也不是本分析的 q20。
https://ops.fhwa.dot.gov/perf_measurement/ucr/documentation.htm

## 验证、资源和停止点

- 输出逐日/小时/道路等级分布、未知原因（可重叠）、支持度变化、相邻窗口转换。
- 全体物化窗口与 study-observed edge union 的无事件窗口分开计数；不是全路网覆盖率。
- 每个 Validation 日选择订单支持窗口最多的两个 edge 作示例，明确非随机代表样本。
- 最小 3 个测试；逐日处理、单进程、CPU、BLAS 单线程、无稠密时空矩阵。
- 30 分钟硬超时，0.2 秒采样 RSS。失败不自动重启分析。
- 命令：`python -m stage3.scripts.traffic_state_batch1`。
- 原 M3 checkpoint、Stage3 profile 执行前后 SHA 一致。
- 完成第 4 步后停止，不进入 profile 适配、E/Q/C 替换或动态对照。
