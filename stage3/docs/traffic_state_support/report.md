# 研究性交通状态：把 UNKNOWN 与低支持度分开

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: authorized descriptive analysis
- Origin Date: 2026-09-27
- Verification Status: ANALYZED（非独立交通真值验证）
- Version Label: traffic_state_support_v1

## 结论

用户提出的担忧成立了一部分：**原 UNKNOWN 把“无法估计”和“有估计但支持不足”混在了一起。**
分离后 UNKNOWN 从预测时间的 21.994% 降至 10.047%，不需要重新匹配、训练或补齐数据。
但恢复部分有 76.464% 的预测时间进入 MIXED，不能把 UNKNOWN 下降解释为三档拥堵标签已经完善。

本批输出研究用状态与独立支持度，不改冻结 profile，不增加通行门控，也没有跑派单。
100 order-days / 5 dates 原规则保留为严格对照；本批没有选新的最低生产支持阈值。

## 1. UNKNOWN 的实际来源

分母统一为 Validation 25–27 日、29,851 条动态完整历史路线的预测 P50 时间。

| 原因 | 全部预测时间占比 |
|---|---:|
| 有参考，仅 order-days<100 | 9.347% |
| 有参考，order-days<100 且 dates<5 | 2.600% |
| 没有可用的 Train edge reference | 10.047% |
| 预测非法、参考数值非法、仅 dates<5 | 本样本均为 0 |
| 原严格规则非 UNKNOWN | 78.006% |

其中前两项合计 11.947 个百分点，允许计算状态但标记 LOW_REFERENCE_SUPPORT。
剩余 10.047% 是在本次 Train 09–24 的有效直接观测参考表中没有对应 directed edge；
不代表 M3 没有输出，也不等同于 OSM 缺路。没有在本轮为其填道路等级均值或默认畅通。
三天残余 UNKNOWN 分别为 10.118%、10.111%、9.911%，不是单日异常造成。

## 2. 新旧状态并列，不覆盖旧产品

| 状态 | 原严格规则时间占比 | 研究性可计算规则时间占比 |
|---|---:|---:|
| 非拥堵 proxy | 60.165% | 62.836% |
| 拥堵 proxy | 2.671% | 2.780% |
| 严重拥堵 proxy | 0.477% | 0.510% |
| MIXED_EVIDENCE | 14.693% | 23.828% |
| UNKNOWN | 21.994% | 10.047% |

研究性可计算规则只移除了样本量准入，不改变 pace ratio / low-speed share 的分类阈值。
有一个有效参考也能给出数值估计，但不能称为稳定、可信或准确；支持度字段始终同行保存。
三档明确标签合计约 66.126%，不是 89.953%：后一个数包括 MIXED。

## 3. MIXED 是什么

研究性 MIXED 合计 23.828% 预测时间，可完全拆成两类：

| 子类型 | 全部预测时间占比 | 可以怎样解释 |
|---|---:|---|
| 相对减速，但低速占比没有达到 0.5 | 9.980% | 相对自身经验参考变慢，但未同时出现高低速暴露 |
| 低速占比达到 0.5，但相对 pace ratio 未达到 1.5 | 13.847% | 绝对低速较多，但未明显慢于自身经验参考 |

后者在新恢复的低支持部分占主导。可能涉及道路本来较慢、参考不稳定等因素，
但本轮没有鉴别具体原因，不能直接命名为红灯、接客、停车漂移或畅通。
这也说明“相对减速”和“绝对低速”不必强制折叠成同一条拥堵严重度轴。

## 4. Train 内部参考稳定性

运行前固定协议 commit `f1af90e`。只用 09–20 有效 order-edge-day 平均 pace，
按 date+order_id 的稳定哈希分两半；同订单日的所有 edge 保持同组。
两个半样本各自计算 q20，然后在未参与参考拟合的 Train 21–24 同一批完成事件 cell 上比较状态。
没有用 Validation 选参考阈值。

下表主表仅列至少 5 个参考日期的道路；样本量是两半合计的 order-days，不是每半数量。

| order-days | 可比较 edge 数 | q20 对称相对差 P50 | P90 | 后四天可比较 cell 数 | 状态分配变化率 |
|---|---:|---:|---:|---:|---:|
| 1–9 | 611 | 11.908% | 36.657% | 1,494 | 11.580% |
| 10–29 | 1,252 | 7.670% | 24.006% | 7,783 | 10.009% |
| 30–99 | 1,613 | 5.555% | 14.219% | 31,395 | 7.460% |
| 100+ | 4,148 | 2.062% | 6.451% | 376,202 | 2.982% |

仅 1–4 日期、1–9 order-days 组：1,301 个 edge 可比较，q20 差 P50=16.618%、P90=53.961%，
1,463 个 cell 的状态变化率 14.764%。1–4 日期、10–29 order-days 只有 1 条 edge、6 个 cell，
不据此得出稳定规律；完整分母在 `train_stability.json`。

总体 419,969 个 joint-valid heldout cells 中 418,343 个有双半参考可比较；1,626 个无法比较。
14,731 个 cell 的分类不同，合并变化率 3.521%。高支持组贡献绝大部分 cell，不能把这个总体值
直接当成低支持组的稳定性。仅有一个半样本包含的 edge 也明确列为不可比较，而非零误差。

这些是参考抽样敏感性与状态重复性，不是真值准确率、置信区间或证明某阈值“足够”。
样本量组之间道路类型/交通分布可能不同，分半也只有一个固定划分，因此不能把组间差异
全归因于样本量。高支持规则有其稳定性依据，但不意味着低支持估计应在研究里全部丢掉。

## 5. 研究用接口

新增目录 `stage3/output/traffic_state_support/`，与原解释层和冻结主产品隔离：

- `tokens_date=YYYYMMDD.parquet`：订单/traversal/有向 edge/顺序、冻结预测、参考样本数和天数、
  `strict_state`、`research_state`、`reference_support`、`unknown_reason`、`mixed_subtype`。
- `routes_date=YYYYMMDD.parquet`：研究性五状态的预测时间占比、合并拥堵最长时长、低支持及无参考时间占比。
- `train_split_reference.parquet`：两半 q20、样本数和参考差。

日期由分区文件名标识，route 表同时带 date。未知仍留在完整路线分母中。
支持度是另一轴，不是新增 rho、cost 或拒绝原因；没有把新状态与同源 crawl/stop 约束叠加。
新 token 产品没有 completed-traversal 真值标签；21–24 的观测只用于独立的参考稳定性诊断，
不混入 Validation 决策字段。

## 6. 验证与资源

三天严格状态 token 计数与上一批逐类完全对账。新旧预测来源及 route 数据 SHA 与既有 manifest 核对；
冻结 profile、M3、原参考、原配置 SHA 前后不变。运行 37.17 秒，0.2 秒采样 RSS 峰值 694.64 MiB，
单进程 CPU、无 GPU、无路网×时间稠密矩阵。未训练、未读 Test31、未派单。

统计解释检查 11/11：Simpson（给逐日及样本组结果）；ecological（不从 edge/cell 推 AV 安全）；
Berkson（参考和路线均为已筛选样本）；collider（条件于有效/可比较，不作因果推断）；
base-rate（明确不同分母）；regression-to-mean（无干预前后效果）；survivorship（保留不可比较分母）；
look-elsewhere（所有预定组公开）；forking-paths（执行前固定、未挑门槛）；
causality 与 reverse-causality（无因果或性能改善声明）。上述限制没有因工程检查而消失。

## 建议的下一步（未执行）

可继续沿“状态×支持度”研究接口推进，不再要求整条路线零 UNKNOWN。
最需要决定的是 MIXED 的两个可解释子类型在研究情景中如何表达，而不是继续提高支持门槛。
可以保留相对减速与绝对低速两个维度，再明确 C/M/A 的适配假设；不要把 23.828% MIXED 全判禁行。
残余无参考约 10.047% 可保留为显式缺失并做情景敏感性，而不是自动当作畅通或最坏拥堵。
任何预算/未知情景规则仍需明确后再做小范围动态对照，本轮未自动启用。

相关文件：`protocol.md`、`decomposition.json`、`train_stability.json`、`summary.json`。
复现命令：`python -m stage3.scripts.traffic_state_support`（全新输出目录；不覆盖已有结果）。
