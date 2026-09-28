# 独立交通研究策略：实现、离线对账与小窗口动态对照完成

后续授权的“离线独立作用 → F/V/VT固定状态 → 预定动态窗口”三阶段也已于2026-09-28完成，
见 [mechanism_report.md](mechanism_report.md)。下文保留首轮M窗口的历史结果。

## Material Passport

academic-research-suite / experiment-agent，2026-09-27。
实现/离线：PASS；动态 retry1：COMPLETE（2026-09-28，退出码0）。
两个条件均为254/379单匹配，全部266条新增物理分配一致；冻结默认值不变。
完整结果和解释见 [window_report.md](window_report.md)，机器摘要见 [window_summary.json](window_summary.json)。
下文首次失败记录保留为历史，不代表当前仍在等待授权。
协议提交 `e13325d`，首次执行代码提交 `f341e83`。

## 已实现

默认 `traffic_research_policy=FROZEN`，不读取新表、不改变原 exposure。
显式 `TRAFFIC_REPLACEMENT_V1` 加上 `traffic_research_table` 和 `traffic_research_budget=0.05` 才启用。
不改变原 request rho、hard_state、冻结 profile、模型、已存 41 场景。

- C：相对减速型 MIXED + 拥堵 + 严重拥堵的预测时间占比<=5%。
- M：严重拥堵占比<=5%。
- A：没有额外已知类别限制；仍受原结构和其他条件约束。
- UNKNOWN/低支持不新增门控，完整路线时间仍是分母，未知占比独立记录。
- dynamic exposure 只用 CV/acceleration 六项，沿用累计 Gamma；移除 crawl/stop 六项。
- 新交通门控与 Gamma 的含义分开，不把 5% 转成新的 rho/cost。
- HV 不受交通门控；fallback 不继承 ORIGINAL 解释，当前 evidence-complete AV 请求均为 ORIGINAL。
- 新损失字段 `gate_av_loss_traffic_policy` 与 Top-K 损失分开，并扩展对应守恒/聚合逻辑。

## 离线对账

Validation 29,851 条与 Test31 29,864 条 complete dynamic ORIGINAL routes；复用冻结预测，不做新推理。
5% 交通条件单独计数如下；不是全条件可派单数，也不是服务率：

| 数据 | C 在预算内 | M 在预算内 | A 在预算内 | 分母 |
|---|---:|---:|---:|---:|
| Validation 25–27 | 13,679 | 29,359 | 29,851 | 29,851 |
| Test31 | 16,983 | 29,616 | 29,864 | 29,864 |

0/5/10% 全部预定离线计数在 `offline_summary.json`，没有根据通过率重选预算。
通过包括未知暴露的路线，语义仅为“已估计环境外暴露未超预算”，不是未知部分已经适配。
新 variability rho 不大于原 12 项 rho，C/M/A 嵌套及预算单调性均通过。
Test31 M3 manifest 的 decision-time/predicted-progression 标志和源 SHA 已核对。
冻结 profile/reference/config 未变。Test31 不是新的未见科学验证集，本轮只用来执行已授权窗口对照。

## 首次动态启动与修复

固定条件是 q50/M/P70、同一原生 10:30 checkpoint、[10:30,10:45) cohort、
冻结策略与 5%研究策略两个条件；SINGLE_SOURCE_MATRIX、30秒滚动、300秒到达耐心、不加 dwell。

首次运行在预检阶段遇到 checkpoint 中 20161101 的未来请求：
`missing selected-route traffic evidence`。原代码误检查了全天所有请求，尽管窗口已经规定不消费10:45以后需求。
错误发生在 routing adapter 创建与第一个 time_trigger 之前，**执行仿真步数为0**；没有分配或服务效果结果。
失败记录保留在 `stage4/output/traffic_research/window/summary.json`，其中 `rows=[]`。

现已修复为只检查本次消费的历史与窗口内请求。只读诊断结果：

- 2,026 个范围内原 AV-eligible 请求有精确 ORIGINAL/date/profile 对应，原 rho 与预测时长一致。
- checkpoint 原动态累计量为 95.64505263388362；按保留 CV/acceleration 定义换算为 61.65063306689326。
- 计数和 static/speed 不改，物理任务不重排；换算只是保证新 Gamma 语义一致。
- 此次诊断仍为0仿真步，不是成功重跑。
- 32 项策略/内核/生命周期/门控/稀疏基线测试通过，compileall 通过。

测试最初遇到系统 pytest 临时目录 ACL 和未设置 FLEETPY_ROOT，已改用新的工作区临时目录和既有固定 FleetPy checkout。
仅设置子进程 Git safe.directory，未改全局 Git 或环境依赖。

首次失败后依据技能的失败不自动重试规则暂停。用户于2026-09-28授权后，已执行以下命令，
结果写入新目录 `retry1`，没有覆盖第一次失败记录：

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:GIT_CONFIG_COUNT='1'
$env:GIT_CONFIG_KEY_0='safe.directory'
$env:GIT_CONFIG_VALUE_0='D:/pycodes/didi_xian_raw/.external/FleetPy'
D:/anaconda/envs/stage0-valhalla/python.exe -u -m stage4.analysis.traffic_research_window --fleetpy-root D:/pycodes/didi_xian_raw/.external/FleetPy --attempt retry1
```

运行目录为本 worktree 根目录。每条件1800秒、单进程CPU、2GiB RSS告警、无GPU。
两个条件完成且原生对账通过。本窗口没有观察到服务差异；未进入全天试验或其他profile。
