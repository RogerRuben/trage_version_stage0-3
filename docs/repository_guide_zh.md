# 代码仓库详细说明与导航

盘点日期：2026-09-14。基于提交 `881152b103f0f976ebff4acd9b06c40d1801dee3`。

本文面向接手项目、定位代码、查找结果和后续维护。内容依据当前文件、配置、接口契约和已提交报告整理；不是新的运行验收报告。本次未重跑生产、训练或实验，未逐个验证大型数据文件完整性，也未测量全盘占用。

## 1. 先确认你打开的是哪个工作区

本机有两个 Git worktree，不能混为一谈：

| 位置 | 分支 / 盘点时 HEAD | 作用与状态 |
| --- | --- | --- |
| `D:/pycodes/didi_xian_raw` | `codex/pipeline-rebaseline` / `fd1e7c0` | 较早的重建工作区；盘点时存在大量未提交删除、`.gitignore` 修改及未跟踪文件。本次不处理这些改动 |
| `D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla` | `codex/stage2-v5-micro-transfer` / `881152b` | 最近 Stage0–4 和 Manuscript V3 的工作主线；开始盘点时干净 |

工作树目录名保留了早期 Stage0 任务名称，分支名保留了 Stage2 任务名称，**都不代表仓库只做到那个阶段**。

后文相对路径均以第二个工作区为根。远端为 `https://github.com/RogerRuben/trage_version_stage0-3.git`。Git worktree 共享仓库对象，但各自有独立工作目录和索引；在根工作区执行 `git add -A` 可能提交大量与本任务无关的删除。

建议先检查：

```powershell
Set-Location D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla
git branch --show-current
git status --short
git worktree list
```

## 2. 这个仓库解决什么问题

代码将西安滴滴 GPS 轨迹及 OSM 路网转化为高质量路线、直接观测动态标签、决策时点预测、AV 路线适配接口，再用于 HV/AV 混合车队滚动派单。当前论文主线是：名义车辆供给不等于状态相关服务能力。

```text
原始轨迹 + OSM PBF / Valhalla tiles（外部、本地数据）
    ↓
Stage0 v6：地图匹配、路线质量、物理 traversal 与直接 interval
    ↓ stage1/input_v1
Stage1 v3：直接动态标签、Train reference/CDF、支持与缺失掩码
    ↓ stage1/output_v3 + models
Stage2 v5.2：冻结 M3 的 decision-time operational prediction
    ↓ 预测、checkpoint、特征和来源契约
Stage3 odd_tod：静态路网/交叉口、能力包络、hard + continuous、有限 fallback
    ↓ 30,000 订单 × C/M/A 三 profile 的固定路线接口
Stage4：FleetPy 状态推进 + 稀疏候选图 + lexicographic assignment
    ↓
41 个正式全天场景 + 独立机制/稳健性分析
    ↓
结果报告、Manuscript V3、附录和图稿计划
```

这不是一个开箱即用的单包应用：仓库含多个研究代际和本地数据依赖。旧 `run_pipeline.py` 是 canonical rebaseline smoke 入口，不是当前所有阶段的最终生产总入口。

## 3. 顶层目录地图

| 路径 | 内容 | 阅读建议 |
| --- | --- | --- |
| `stage0/` | 多代轨迹处理、路网、匹配和生产 | 优先 `v6/`、最终配置与契约 |
| `stage1/` | 动态标签、reference/CDF、聚合和质量 | 优先 `v3/` |
| `stage2/` | baseline、深度模型、transfer、审计 | 优先 `v5_2/`；保留 v4/v5 依赖 |
| `stage3/` | 历史 condition-vector 路线及当前 ODD/TOD 接口 | 优先 `odd_tod/` 与 `docs/odd_tod/final/` |
| `stage35/` | 早期离线路线产品的定义与契约 | 不是当前新增的独立生产阶段 |
| `stage4/` | 旧仿真、当前 FleetPy、派单、分析、论文 | 优先 `fleetpy_adapter/`、`dispatch/`、`analysis/` |
| `canonical_pipeline/` | 早期统一 manifest/preflight/registry 治理 | 历史 smoke 体系，不能代替新阶段契约 |
| `config/` | 早期全链路配置和 manifest schema | 当前配置主要在各 stage 下 |
| `artifacts/` | canonical / exploratory / deprecated 分类及登记 | 不等于包含所有完整实验 payload |
| `docs/pipeline_contract/` | 早期全链路字段与阶段契约 | 与最终分阶段契约冲突时须追溯版本 |
| `docs/pipeline_rebaseline/` | 历史重建审计和人工 truth 资料 | 用于溯源，不当作当前默认配置 |
| `scripts/`、`tests/` | 早期全局审计和测试 | 不覆盖全部较新分阶段测试 |
| `.github/` | GitHub 配置 | 不能仅凭其存在认定全链路 CI 已通过 |

### 3.1 规模快照

以下为盘点基线的 `git ls-files` 统计，**不是磁盘大小、代码行数或本地全部文件数**。

| 区域 | 跟踪文件 | Python 文件 | Markdown 文件 |
| --- | ---: | ---: | ---: |
| Stage0 | 158 | 119 | 24 |
| Stage1 | 62 | 34 | 11 |
| Stage2 | 482 | 215 | 44 |
| Stage3 | 130 | 48 | 37 |
| Stage3.5 | 4 | 0 | 2 |
| Stage4 | 520 | 162 | 100 |
| 顶层 docs | 85 | 0 | 21 |

较多文件来自历史版本、报告、配置和测试，不能据目录名或文件数量判断当前执行量。

## 4. Stage0：轨迹匹配和高质量订单生产

主要实现位于 [stage0/v6](../stage0/v6)，入口为 [cli.py](../stage0/v6/cli.py)，最终配置为 [stage0_v6_final.yaml](../stage0/config/stage0_v6_final.yaml)。

| 模块 | 主要职责 |
| --- | --- |
| `parser.py`、`coordinates.py` | 原始记录与坐标处理 |
| `preprocess.py`、`eligibility.py` | 预处理、有效性和低成本预筛 |
| `valhalla_client.py`、`build_tiles.py` | Valhalla 调用和 tiles 构建 |
| `canonical_mapper.py` | 路网 identity 与 canonical 映射 |
| `order_processor.py`、`products.py` | 单订单处理与产品组织 |
| `quality.py`、`final_quality.py` | 质量状态与最终几何/区段判断 |
| `automated_audit.py`、`manual_audit.py` | 自动审计、人工审查材料 |
| `stage1_production.py` | 分日、分 bucket 的候选选择和核心数据生产 |
| `pipeline.py`、`final_report.py` | 流程与报告 |

GPS、route、dynamic、canonical 是独立状态。不能用动态标签稀疏直接判路线错误，也不能把历史反向行驶简单改成当前道路正向 identity。地图匹配解释历史观测与 AV 可路由性判断属于不同层次。

### 4.1 下游产品是什么形式

核心目录 `stage1/input_v1/` 由 Stage0 生产，按 split/date/bucket 分区。它不是“每条 link 都有精确出入时间”的单一表：

| 产品 | 粒度 / 含义 | 使用注意 |
| --- | --- | --- |
| `order_base` | 订单及四类质量状态 | 只有核心合格订单进入主产品 |
| `route_parts` | 有向 canonical route sequence | 用路线顺序字段排序，不靠文件行顺序 |
| `route_segments` | 连续区段及区段质量 | 局部与整体质量分别表达 |
| `link_traversals` | 一次物理连续 link 访问 | 物理距离不得按每个 GPS interval 重复计入 |
| `link_interval_observations` | 直接可计时 GPS interval | 只有 `direct_observed` 且 `label_valid=true` 可作直接标签 |
| `interval_measurements` | direct / supported / interpolated / unresolved 分类 | 非 direct 不能被解释成 observed time |
| `turn_movements` | movement identity 和 provenance | 不默认具有可信 movement delay |
| `quality` | GPS / route / dynamic / canonical 质量 | 不合并为单个真假标签 |

同一个 traversal 可以有多个直接 interval；区间支持或引擎插值不意味着每条经过道路都被直接计时。订单序列、物理 traversal 和动态 observation 必须按标识关联。

### 4.2 数据范围与冻结

[Stage1 输入覆盖报告](../stage1/docs/stage1_input_coverage_report.md)记录的生产目标/合格规模为 Train 20161009–24 共 160,000 单，Validation 25–27 共 30,000 单，Test 31 日共 30,000 单，总计 220,000 单。该结论来自已有报告，本次未重新遍历所有 Parquet 对账。

权威阅读入口：

- [Stage0→1 契约](../stage0/docs/stage0_to_stage1_contract.md)
- [Stage0 冻结清单](../stage0/docs/stage0_v6_freeze_manifest.json)

Stage0 已冻结；只有丢失/重复、守恒错误、大规模系统性错配、下游无法读取 schema 等限定原因允许重开，不因论文效果或覆盖不足继续调匹配。

## 5. Stage1：直接观测动态标签

主实现 [stage1/v3](../stage1/v3)，配置 [stage1_label_schema_v3.json](../stage1/config/stage1_label_schema_v3.json)，主要输出 `stage1/output_v3/`，模型与 reference 在 `stage1/models/`。

| 模块 | 职责 |
| --- | --- |
| `input_adapter.py`、`io.py`、`schema.py` | 输入、分区 I/O、schema |
| `primitives.py`、`aggregation.py` | 原始动态量与聚合 |
| `references.py`、`histograms.py`、`support.py` | Train reference、CDF 与支持统计 |
| `pipeline.py`、`models.py` | fit/transform 流程和模型对象 |
| `preflight.py`、`audit.py`、`scientific_review.py` | 工程验证与科学分布诊断 |
| `freeze.py` | 发布身份和冻结 |

Stage2 的主要监督单位是 `(order_id, traversal_id)`。关键动态量包括 crawl、stop、bounded speed CV、bounded acceleration RMS；raw 值和 percentile 是不同语义，不能互换。

必须保留缺失掩码与 unavailable reason，不能把缺失标签补成 0。reference/CDF 在训练范围拟合，不能借 Validation/Test 拟合。后续 Stage2 rolling 划分也不能把用更晚日期拟合的 percentile 当无泄漏主要目标。

阅读 [Stage1→2 契约](../stage1/docs/stage1_to_stage2_contract.md)。CLI 提供 `fit`、`transform`、`verify`、`preflight`、`scientific-review`，其中验证/报告命令也可能写报告，并非全部零副作用。

## 6. Stage2：预测模型与冻结选择

当前决策时点预测主线位于 [stage2/v5_2](../stage2/v5_2)，配置 [stage2_v5_2.json](../stage2/config/stage2_v5_2.json)。

| 模块组 | 职责 |
| --- | --- |
| `transfer_data.py`、`feature_binding.py`、`protocols.py` | 数据、特征绑定和日期/角色协议 |
| `structure_features.py`、`temporal_adapter.py` | 结构与时间特征 |
| `models/rc_mstnet_transfer.py`、`training.py` | 模型与训练 |
| `evaluation.py`、`micro_metrics.py`、`micro_products.py` | 评估和微观产品 |
| `support_transfer.py`、`sparsity_support.py` | 支持信息与 transfer 相关逻辑 |
| `b1_freeze.py`、`phase_c_report.py`、`final_freeze.py` | 阶段选择与最终冻结 |
| `sparsity_diagnostic.py`、`upstream_sampling_audit.py` | 时空稀疏和上游抽样诊断 |
| `verification.py`、各 schema audit、`performance.py` | 契约、正确性和性能边界 |

### 6.1 最终采用哪个模型

[最终 Stage2→3 契约](../stage2/docs/v5_2/stage2_v5_2_to_stage3_contract.md)明确：

- 最终工程预测器为 **M3 / structured_representation**。
- checkpoint 路径是 `stage2/output_v5_2/development/M3/epoch_004.pt`；文件存在、内容 SHA 和可加载性应在推理前按契约验证。
- 下游可用量为 `travel_time_p50`、`pace_p50`、`crawl`、`stop`、`speed_cv`、`acceleration_rms`，以及必要身份/支持/完整性字段。
- RTS 仅诊断，不进入 AV feasibility、ODD/TOD、fallback 或 Stage4。
- M4 没有通过继续采用规则；M5/M6、Phase D、Transfer-v2 不应因目录或 CLI 仍保留而被视为获准执行。

上述契约内的“Stage3 尚未授权”是当时的历史阶段状态，已被后续用户授权和 Stage3/4 产品推进所超越；**不可把每份历史报告的下一阶段授权字段当成现在的总状态**。

### 6.2 为什么不能直接删 v4/v5

已检查到 `v5_2` 导入 `stage2.v5.shards`、`stage2.v5.data`、`stage2.v5.models.rc_mstnet_v5` 和 `stage2.v4.models.baselines`；Stage3 推理也导入这些基础组件。因此旧目录中既有被替代实验，也有当前共享依赖。“版本旧”不等于“未使用”。

`output_v4`、`output_v5`、`output_v5_1`、`output_v5_2` 是不同代际产物，不应自动拼接。新的 fold 要遵守自己的训练/校准边界，不能仅按目录中已有日期取数据。

## 7. Stage3：路线 operational suitability 接口

当前主实现 [stage3/odd_tod](../stage3/odd_tod)，不是早期 `stage3/scripts/train_*` 的 condition-vector 流程。

| 文件 | 阶段职责 |
| --- | --- |
| `static_inventory.py` | OSM/maxspeed/signals/restrictions/layers 等静态数据盘点 |
| `network_foundation.py` | 网络与 speed-domain 基础 |
| `intersection_complex.py`、`s2b11_tolerance_closure.py`、`s2b2_final_freeze.py` | 交叉口 complex、5m/10m 审查和冻结 |
| `capability_envelope.py` | 静态/动态能力包络、训练校准 |
| `s4_inference.py` | 冻结 M3 推理 |
| `original_route_suitability.py`、`operational_suitability.py` | 原路线评估及 hard + continuous 输出 |
| `finalization.py` | 有限 fallback 与最终 Stage4 接口 |

最终 [Stage3→4 契约](../stage3/docs/odd_tod/final/stage3_to_stage4_contract.md)指向：

```text
stage3/output/odd_tod/final/
  test31_stage3_to_stage4_interface.parquet
  test31_fallback_route_edges.parquet
  test31_fallback_candidates.parquet
  stage3_final_summary.json
```

本次确认上述文件名在本地目录存在；不等于重新校验全部内容。接口规模由契约规定为 30,000 订单 × 三 profile，即 90,000 订单-profile 行。

核心语义：

- `hard_state` 为 FEASIBLE / UNKNOWN / INFEASIBLE；未知不是确定不可行。
- `rho_static`、`rho_dynamic`、`rho_speed` 保持独立，不是事故概率，也不是统一安全分。
- 静态 D 是 boundary incoming/outgoing road-class diversity，不是内部 edge diversity。
- route reference 指向 ORIGINAL 或冻结 FALLBACK；Stage4 不自行重选服务路线。
- 有限 K=1 搜索未建立可行路线，不代表 OD 上绝对不存在可行路线。
- passenger acceptance 属于 Stage4，不由 Stage3 预测。

旧 [stage35/README.md](../stage35/README.md)描述的离线路线层概念仍有历史意义，但实际最新接口由 `odd_tod/finalization.py` 和上述最终契约提供；不要额外启动一套 Stage3.5 生产。

## 8. Stage4：FleetPy、派单与实验

### 8.1 当前执行层

| 路径 | 职责 |
| --- | --- |
| `fleetpy_adapter/upstream.py`、`test31_demand_adapter.py` | 上游接口和需求接入 |
| `native_demand.py`、`native_network.py`、`native_simulation.py` | FleetPy 原生需求/路网/仿真适配 |
| `mixed_fleet_adapter.py`、`native_fleet_control.py` | 混合车队和控制连接 |
| `replay_service_time_adapter.py`、`valhalla_time_adapter.py` | 服务时长与 pickup routing |
| `dispatch/candidate_graph.py` | 稀疏候选生成 |
| `dispatch/acceptance.py`、`gate_diagnostics.py` | 接受门控与各层诊断 |
| `dispatch/solver.py` | 稀疏 lexicographic 求解 |
| `dispatch/exposure.py` | AV 系统累计 exposure 状态 |
| `dispatch/rolling_or_control.py`、`odd_aware_runner.py` | patience-aware rolling 与 ODD 决策核 |
| `dispatch/fleet_normalization.py`、`vehicle_hour_normalization_correction.py` | 车辆活跃小时口径 |
| `dispatch/final_experiment_runner.py`、`final_experiment_aggregation.py` | 注册场景执行与汇总 |

### 8.2 当前模型的几个不能写错的点

1. 决策周期 30 秒、基准 pickup patience 300 秒；实际出发/boarding proxy 被用作零 lead request release，是假设，不是观测到的真实叫车时刻。
2. HV 继承 empirical session，AV 为 full-horizon availability；车辆类型与可用性策略要分开理解。
3. `q_A = H_AV / H_base` 是基准归一化 AV 活跃小时水平，不必等于场景内部 AV 小时份额。
4. 目标优先级为 critical → total → carry-over → pickup ETA → 可选 cost。
5. Gamma 约束控制全系统累计 AV assignment 的**平均** family exposure，不是每车累计预算；epsilon cost stage 限制 aggregate pickup ETA 退化，不放松前面服务计数最优值。
6. Stage4 不拥有服务路线决策变量；routing pickup 不等于在线重规划服务路线。
7. `solver.py`、FleetPy 与 route evidence 各负责不同层次；切勿把 unknown、passenger rejection、pickup failure 和未被匹配合成一个理由。

### 8.3 正式结果在哪里

- 场景登记：`stage4/config/experimental_design/scenario_registry.csv`。
- 执行配置：[final_experiment_execution.json](../stage4/config/final_experiment_execution.json)。
- 正式输出：`stage4/output/final_experiments/`。
- 完成报告：[stage4_final_experiment_execution_summary.md](../stage4/docs/final_experiments/stage4_final_experiment_execution_summary.md)。

已有报告记录 41/41 个唯一场景完成、失败 0；分组行数为 MAIN/BENCHMARK/ODD/COST = 27/4/3/8，其中有复用行，不能把 42 个分组行误解成 42 个独立执行。历史运行耗时 16.647 小时、观测峰值 RSS 1674.7 MB 是该轮报告数字，不是当前进程监控或未来资源保证。

### 8.4 当前分析代码

| 文件 | 分析主题 |
| --- | --- |
| `analysis/result_analysis.py` | 正式场景结果分析 |
| `prospective_gate_analysis.py`、`paper_enhancement_gate_decomposition.py` | 前瞻门控、edge funnel 口径 |
| `mechanism_validity.py` | neutral identity 与 E/U/M matching-capacity 机制 |
| `recover_rt_parameters.py`、`rt_patience_sensitivity.py` | RT 参数恢复及固定状态 timing 敏感性 |
| `frozen_state_prediction_ablation.py` | 冻结状态 prediction relevance |
| `repositioning_robustness_analysis.py`、`efficient_routing_evidence.py` | 重定位与 routing 效率证据 |
| `fleet_spatial_representativeness.py` | 车队空间代表性 |

`dispatch/` 中另有 deterministic / efficient routing 和 repositioning runner。这些是独立研究扩展，不能默认替换 41 个正式场景的既定策略。旧 `core/`、`simulator_v3/`、`evaluation/`、大量 `scripts/` 仍用于历史实现或复现，不能把它们和当前主 runner 随意混用。

## 9. 论文与科学证据导航

当前最新完整稿是 [Manuscript V3](../stage4/docs/manuscript_v3/stage4_manuscript_v3.md)，其入口 [README](../stage4/docs/manuscript_v3/README.md)关联附录、文献核查、五图计划和修订报告。

| 目录 | 定位 |
| --- | --- |
| `stage4/docs/paper_draft/` | 较早 Results/Discussion 分段稿 |
| `stage4/docs/full_paper/` | 较早完整母稿 |
| `stage4/docs/paper_redesign/` | 理论笔记、机制报告、RT 恢复与多代 storyline |
| `stage4/docs/manuscript_v3/` | 当前已提交完整正文、附录和五图设计 |
| `stage4/output/paper_enhancement/` | 机制、routing、重定位等分析产物 |

当前证据应分层阅读：

- 全天服务结果：41 场景，受 zero-lead/300s-patience 设定限定。
- 固定状态 AV 子图：E 是边数、U 是有选项订单数、M 是最大匹配数。不是全混合车队日服务的分解；多状态求和也不是全天唯一订单。
- RT 恢复：fingerprint-verified reconstruction，不是找回原 manifest 或逐订单原 hash。
- prediction：DECISION-RELEVANT；没有 common independent evaluator，不能改称 decision-superior。

用户随后提出的 V3.1 capacity-substitution / Hall-deficiency 理论强化、closest literature 扩充和最终作图，在当前 `881152b` 尚未落入完整 V3.1 稿。上轮仅讨论了方案，本次仓库说明不把计划写成已完成结果。

## 10. Git 保存了什么，没保存什么

`.gitignore` 默认忽略 raw 数据、地图/tiles、多数 `output*`、`stage1/input_v1`、`stage1/models`、Stage4 data/input，以及 Python 缓存。

但 `.gitignore` 不会停止跟踪已提交文件：盘点时 Stage2 输出目录下有 3 个跟踪文件，Stage4 输出目录下有 93 个跟踪文件。因此：

- Git clone 不等于完整研究数据恢复。
- 目录被 ignore 不等于其中每个文件都未提交。
- manifest 存在不代表其引用的 payload 一定存在。
- 不要对 output 目录统一 `git add -f`，可能上传体积大或含订单/司机标识的产品。
- 备份和清理前应分别查 Git 跟踪状态、配置依赖、manifest 引用和可复现成本。

本地大数据候选位置包括 `stage1/input_v1`、`stage1/output_v3`、各 `stage2/output_*`、`stage3/output`、Stage4 运行日志/缓存、根目录原始轨迹和 tiles。本次不提供未经扫描的 GB 数字，也不把这些目录列为可直接删除清单。

## 11. 环境与外部依赖

- 历史主要运行环境为 Conda `stage0-valhalla`；环境文档记录 Python 3.12.13 与 pyvalhalla 3.8.2。
- 原始 PBF 和 Valhalla tiles 位于主工作树之外的共享数据路径，例如 `D:/pycodes/didi_xian_raw/map_data`、`D:/pycodes/didi_xian_raw/valhalla_data`。
- 本机 `.external/FleetPy` 目录存在。固定依赖记录见 [fleetpy_dependency.md](../stage4/docs/fleetpy_spike/fleetpy_dependency.md)，指定 SHA `0379f9725a147ff33c674de4884cdf89fd787fa9`；本次未重新验证该外部 checkout 的 HEAD。
- Stage2 使用深度学习组件；Stage4 当前正式配置明确 CPU-only / sparse CSR，无 GPU。
- 仓库存在 `stage0/requirements.txt` 和 `stage0/v6/requirements.txt`，不应当作整条 Stage0–4 主线的统一锁文件。各历史版本记录、NumPy 兼容性和实际环境可能不同，禁止为了“整理”直接在 base 环境全量升级或安装历史 pin。

完整复现至少要匹配代码、配置、原始数据、PBF/tiles、Stage0/1 产品、模型/特征/CDF、Stage3 interface、FleetPy commit 和场景 registry。只复制本仓库源码不够。

## 12. 命令导航与测试边界

以下是经源码确认的入口，**列出不等于授权重跑**。建议先看 `--help`，再按阶段报告构造命令；导入阶段仍可能因缺少依赖失败。

```powershell
conda run -n stage0-valhalla python -m stage0.v6.cli --help
conda run -n stage0-valhalla python -m stage1.v3.cli --help
conda run -n stage0-valhalla python -m stage2.v5_2.cli --help
conda run -n stage0-valhalla python -m stage3.odd_tod.finalization --help
conda run -n stage0-valhalla python -m stage4.dispatch.final_experiment_runner --help
```

| 入口 | 真实操作 | 风险提示 |
| --- | --- | --- |
| Stage0 `final-600` | 固定样本处理/报告 | 会运行匹配，不是只读 |
| Stage0 `build-stage1-input --resume` | 全量核心生产 | 不应为修改下游模型而重跑 |
| Stage0 `verify-stage1-input` | 产品校验 | 全量 I/O 仍有成本 |
| Stage1 `fit` / `transform` / `verify` | reference 拟合、标签生成、验证 | 配置、model-root 与冻结身份必须匹配 |
| Stage2 `train-model` / `evaluate-model` | 训练/推理 | CLI 能调用不代表阶段已经授权 |
| Stage3 `finalization` | 最终接口/fallback | 不应使用 `--force` 随意覆盖冻结产品 |
| Stage4 `run-one` / `run-batch` | 单场景/批次 | 必须显式 FleetPy 路径、场景或 phase；不是新的自动全链路入口 |

根 `pytest.ini` 默认只列出 `tests`、`stage0/tests`、`stage1/tests`、`stage2/tests`、`stage4/tests`。它**没有显式覆盖** `stage2/tests_v4`、`tests_v5`、`tests_v5_2`、`stage3/tests_odd_tod`、`stage3/tests_v5_1`。因此一次裸 `pytest` 通过不能叫全部当前阶段测试通过。

专项测试需选对应目录；`compileall` 只查编译语法，`git diff --check` 只查差异格式，二者都不证明科学正确性。本次仅文档整理，不运行任何测试/实验，也不刷新冻结 QA 状态。

## 13. 内存、存储和维护规则

当前正式执行配置包含 `max_parallel_scenarios=2`、启动前剩余 RAM 门槛 4 GB、单场景 7200s guard、CPU-only。它们是既有配置，不是建议所有机器照抄的安全保证。另一些后续任务要求单进程，实际执行要服从具体协议。

维护时优先遵循：

1. 按 date/bucket/场景读取，避免汇总全日期的大 DataFrame。
2. 保留稀疏候选、Top-K 和缓存边界，不构建订单数 × 车辆数的稠密大矩阵。
3. 路由通常比 solver 贵，不为求解器引入 GPU 或无依据的优化重构。
4. 不把目录叫 cache 当作可删依据；预测缓存可能是冻结结果证据。
5. 不随意重建 tiles，不删除 checkpoint、reference/CDF 或冻结输入。
6. 修改只覆盖相关模块；检查跨版本 import 后再谈迁移/废弃。
7. 清理文件需另行确认精确目标，本说明不授权删除。

## 14. 推荐接手顺序

第一遍只理解主线：本文 → Stage0→1 → Stage1→2 → Stage2 v5.2→3 → Stage3 final→4 → Stage4 正式实验报告 → Manuscript V3。

第二遍按工作目标读代码：

| 要做什么 | 从哪里开始 |
| --- | --- |
| 理解订单/link/interval 的组织 | `stage0/v6/products.py`、Stage0→1 契约 |
| 查标签缺失、支持与 reference | `stage1/v3/`、Stage1→2 契约 |
| 查预测泄漏或 M3 输入 | `stage2/v5_2/transfer_data.py`、`stage3/odd_tod/s4_inference.py` |
| 查 AV route readiness / rho | `stage3/odd_tod/operational_suitability.py`、`finalization.py` |
| 查派单为何没选某条 arc | `candidate_graph.py`、`gate_diagnostics.py`、`solver.py` |
| 查 nominal hours / availability | Stage4 normalization 模块和对应修正报告 |
| 查论文数字 | 正式实验汇总 + `docs/paper_redesign/` 的专题报告 |
| 改论文而不动结果 | `docs/manuscript_v3/`；保留条件解释与版本记录 |

判断“当前权威版本”时，以明确绑定来源的最终接口、后续闭环报告和调用链为准，而不是 README 时间、目录编号大小或某份旧文档中的 PASS/NO 字段。

## 15. 本次整理完成了什么

新增本说明，并更新根 README 为当前主线导航。没有移动、删除、重命名源文件；没有修改原始数据、配置阈值、训练 checkpoint、生产结果或实验逻辑。旧根工作区已有删除保持原样。后续若要物理目录重构或空间清理，应先另列精确依赖和目标，再单独执行。
