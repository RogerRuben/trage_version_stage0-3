# C/M/A共同状态交通开关对照：执行前协议

用户于2026-09-28授权完成下一步，包含统一关闭累计Gamma。基线dff7e69。

- 两个时点10:30/17:30，每个15分钟新需求+300秒耐心、已接任务排空；C/M/A×OFF/ON=12条件。
- 共同物理起点复用已验证M（ODD_Q50_M_P70_REFERENCE）的对应原生checkpoint；
  不复用C/A各自的历史状态，不重算历史派单。结论条件于这一共同M历史，并非profile-neutral历史。
- 车辆身份、位置、native任务、availability政策、已分配/完成/过期状态、等待订单搜索历史全部保持。
- 按order_id更新请求的profile、hard_state、evidence、selected_route_type及各rho；
  需求时间、坐标、realized/predicted service time必须逐单一致；保留native对象引用。
- passenger hash/rate/seed、成本、TopK、路由、耐心、无重定位/无额外dwell设定继承同一源。
- profile间既有结构/证据资格差异仍保留；跨profile总差异不是仅由交通状态造成。
  每个profile内ON−OFF才是本批交通门控的条件性模拟对照。
- OFF=VARIABILITY_ONLY_V1，ON=TRAFFIC_REPLACEMENT_V1。共同保留CV/acceleration六项；
  全部Gamma=null，因此这些连续暴露不作累计准入约束。其他结构/证据门控保持原样。
- 5%和UNKNOWN规则不变。C相对MIXED+congested+severe；M severe；A已知类别无额外限制。
- 历史exposure账本保留M原值（不可解释为C/A累计暴露）；Gamma关闭时不影响准入。
  不用新profile倒推否决历史任务；仅报告新服务cohort的交通暴露。
- A OFF/ON应逐物理分配一致；否则停止调查，不按结果改参重跑。
- 输出服务数、AV/HV、订单转移、等待、已知环境外/UNKNOWN暴露、路由失败/资源/共同起点QA。
- 源checkpoint/配置/profile/M3保护SHA；原41场景不变；全程单进程CPU，线程1，无GPU/稠密矩阵。
- 每条件1800秒超时，2GiB内存告警；异常落盘并停止，不自动重试。
- 结果独立写stage4/output/traffic_common_state_v1；摘要和报告在本目录。完成后停止，不追加全天/预算搜索。

命令：`python -m stage4.analysis.traffic_common_state --fleetpy-root D:/pycodes/didi_xian_raw/.external/FleetPy`
