# v4 M 全天执行与问题回答范围

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent。
- Origin Mode: 用户授权正式全天执行，复用既有MYOPIC。
- Origin Date: 2026-10-06，Asia/Singapore。
- Verification Status: UNVERIFIED（执行前协议）。
- Version Label: acceleration_v4_full_day。

## 唯一新增执行

20161031、M、既有qA=0.5/P70车队、SERVICE_PRESERVING_LOOKAHEAD，从干净native
状态执行一次。MYOPIC复用已完成结果，不重跑；不新增C/A、日期、参数搜索或
独立性能Gate，不改M3、beta、remaining model、Top-K20/K5、30秒rolling、
300秒patience、600秒时域、三Train情景、10秒求解上限及质量逻辑。

```powershell
python -X faulthandler -m stage4.analysis.symmetric_flexibility_full_day `
  --fleetpy-root D:/pycodes/didi_xian_raw/.external/FleetPy `
  --policy SERVICE_PRESERVING_LOOKAHEAD `
  --acceleration-config stage4/config/symmetric_flexibility_acceleration_v4.json
```

正式运行前只修可选磁盘缓存的初始化/关闭故障降级并执行已有相关回归。
缓存失效只丢缓存，不丢候选或结果；raw内存50,000项、磁盘512MiB上限不变。
增加缓存淘汰/错误诊断以便解释真实瓶颈，不增加科学约束。
两个路由进程，OMP/OpenBLAS/MKL/Arrow线程1，CUDA禁用；不构建稠密矩阵。
保留原10800秒行政上限和2048MiB进程组上限。运行异常时保存产物，不静默
重试/改阈值。代码在启动前提交；运行中不修改加载的核心实现。

## 回答问题

1. 全天服务增益：相同30,000源订单/28,367共同订单对账，服务数、两个分母的
   服务率、gained/lost和按小时净变化。单日期/单seed内部比较，不作因果/显著性声明。
2. 等待代价：总体已服务均值/分位数，同时给共同已服务订单配对差异，避免
   将不同被服务人群的均值差误作每个人等待变化。
3. AV/HV分工：分类型服务数、同单HV/AV切换、接驾/载客时间及按小时变化。
   AV服务份额不等于baseline-normalized active-hour ratio。
4. 性能：实际总耗时、routing/候选/未来图/纯优化分项、raw计算与缓存复用、
   淘汰/可选缓存错误、模型规模、回退及RSS。比较v2时承认机器负载/并列最优差异。
5. C/M/A稳定性：本次M一场不能证明。报告明确未识别，不将其补成假结论；
   不为此自行启动未授权的C/A全天。

## 输出和停止

主产物：`stage4/output/symmetric_flexibility_v1/accelerated_v4_full_day/SERVICE_PRESERVING_LOOKAHEAD/`。
轻量汇总：本目录自动summary/analysis，加`full_day_findings.json`与
`full_day_report.md`；日志在现有runner_logs下独立v4目录。
完成后只分析上述问题、提交/推送结果，不继续新一轮算法优化或全场景网格。
