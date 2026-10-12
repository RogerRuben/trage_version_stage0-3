# 三种上线位置的固定全天比较

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent。
- Origin Mode: run，位置接入与三组顺序执行。
- Origin Date: 2026-10-12，Asia/Singapore。
- Verification Status: INTEGRATION_CHECKED_RESULTS_PENDING。
- 基线代码：`50a842aea14019a1d00c80b195e87c6c86ad2998`；生产commit由运行绑定记录。

## 问题与唯一干预

在相同C能力、AV基线归一化车时水平约0.10、P70接受情景、相同车辆接单窗口、相同300秒接单/独立600秒接驾规则下，上线位置从全部需求热点改为C兼容首单热点，再改为考虑M3预计时间/落点、后续接续和HV竞争的联合位置，能否增加C有效载客利用，并保持全队服务—空驶取舍？

只读上轮已完成的三份布局，不再生成或调布局。三组统一使用基于20161009–24历史需求分布的通用空车调位，以及当前最高服务目标的同一最优面内空驶优化；全部为`SERVICE_PRESERVING`控制。不是“简单位置＋简单派单”对“联合位置＋新派单”的捆绑比较。

| 顺序 | 组名 | 唯一位置模式 |
|---|---|---|
| 1 | ALL_HOT_COMMON_CONTROL | all_hot |
| 2 | COMPATIBLE_HOT_COMMON_CONTROL | compatible_hot |
| 3 | OPPORTUNITY_JOINT_COMMON_CONTROL | opportunity_joint |

同一布局容器SHA256为`4f11b259103014a0ad03cf62951893279d366abc628f977d28e42b8a9bc23826`，每组851个AV单元、47批；其中236个短窗口在三组仍共用热点规则，联合布局保留六轮受限规划与整数恢复近似，不能称全局最优。

## 不变条件

- Test31相同28,367个请求；上车时间作为释放代理的研究假设不变。接受抽样、能力信息、M3、历史情景、剩余服务模型和类型标签保持。
- 30秒滚动、Top-K20、当前稀疏图与原实时10秒限制保持，不增加AV优先奖励、成本权重或公平性指标。
- 通用调位15分钟更新、2km邻域、最多50次、Top-3、接驾近似300秒及10秒可选调位预算保持；此调位限制不等于乘客600秒接驾上限。
- 各组独立新进程和128 MiB有界磁盘路由缓存，不并行运行，不建立稠密点对矩阵，不使用GPU。旧布局和已完成全天结果不改写、不替代本轮任何一组。
- 每组1536 MiB RSS上限、启动可用内存至少1536 MiB、10800秒总行政时限。受限可选调位有墙钟截断，分析需诚实保留其计数；不声称逐动作严格重复性。
- 失败现场保留、不自动重试。显式`--resume`只跳过完整同代码/配置/输入组，不能恢复半成品仿真状态。

## 已完成的必要接入检查

7项针对性测试通过（新增位置协议/注入两项，复用原协议与结果对账）；编译及diff检查通过。用真实预定供给模板分别注入三份位置：三次均为851个AV，标签SHA相同，原精确接单窗口及HV位置不变。该检查无FleetPy运行、路由或新M3推理。预检查峰值171.35 MiB，可用内存约3493 MiB。

## 本轮命令与产物

```powershell
D:/anaconda/envs/stage0-valhalla/python.exe -u -m stage4.analysis.scheme_a_two_scale_experiment --root . --config stage4/config/scheme_a_layout_comparison_300_600_v1.json
D:/anaconda/envs/stage0-valhalla/python.exe -u -m stage4.analysis.scheme_a_layout_results --root .
```

工作目录：`D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla`。沿用已有轻量协调器与物理回放，不开发另一套仿真器或嵌套evidence体系。前三组完成之后才执行只读结果分析。

原生完整产物保留本地忽略目录`stage4/output/capability_chain_planning/layout_comparison_300_600_v1/`，含逐组summary、订单/派单/移动/求解与分项耗时、子进程退出记录。有界日志与逐5分钟历史时间进度持续落盘。远端只提交代码、配置、聚合摘要与报告，不上传车辆/订单个体明细。

## 分析与停止点

主读数：全队完成/过期、C服务量、C班次内载客/接驾/调位/未占用车时、接驾与调位空驶、已服务等待。逐小时和共同服务订单配对用于解释净差值，不能把候选次数当作额外可服务订单或把忙时全算载客收益。

相同请求释放、C可服务属性与接受抽样必须对齐；班次内活动分区、物理时间链、重复订单与车辆任务重叠按已有读数核对。只读投影、按组释放大表，汇总后不保留原始个体标识/坐标。单日单种子，不进行显著性或因果普遍性包装。

如果联合位置没有改善，也保留结果；不扩大路径轮数、调等待阈值、换窗口或补一轮使其胜出。**完成三组、解释取舍、提交推送后停止；不自动追加M/A、规模、接受率或成本实验。**
