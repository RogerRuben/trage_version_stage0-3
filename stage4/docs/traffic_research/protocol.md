# 独立交通状态替代策略：离线对账与小窗口对照

## Material Passport

academic-research-suite / experiment-agent；2026-09-27；执行前 UNVERIFIED。
用户已授权实现、离线对账与小范围动态对照。冻结 41 场景及 Stage3 profile 不覆盖。

## 策略

默认 FROZEN，不加载研究表。显式 TRAFFIC_REPLACEMENT_V1 才启用：
C 环境外=相对减速型 MIXED+拥堵+严重拥堵；M=严重拥堵；A=0。
背景低速型 MIXED 对三者均不增加暴露。完整预测 P50 时间为分母。
主预算 5%，离线敏感性 0/10%。未知、低参考支持不增加拒绝或惩罚，但必须单独报告。
5% 是用户确认的情景假设，不是安全标准或通过率拟合。

Stage4 实际用累计 Gamma 约束，而非逐路线 12 项均通过。新策略保留这套逻辑，
将 dynamic rho 改为 CV/acceleration 六项最大 utilization，去掉 crawl/stop 六项；
额外仅按已知环境外暴露是否超过 5% 过滤 AV 候选，HV 不受影响。
静态/速度/方向、乘客接受、费用、耐心、Top-K 不变；不改 request 的原 rho 或 hard_state。
该研究门控不等于结构性 hard infeasibility。统计单独记录，不能混入 evidence 或 Top-K 损失。
仅支持 ORIGINAL 且具备动态证据的路线；本次所有 frozen evidence-complete AV 路线都是 ORIGINAL。
无证据/fallback 保留基线不合格状态，不用原路线交通信息冒充 fallback。
若未来出现 evidence-complete fallback，本策略须明确报错，不能静默继承 ORIGINAL。

## 离线

复用 Validation 25–27 的 research token 表，严格禁止 realized labels 进入策略。
用同样定义派生已有 Test31 冻结预测，仅服务本次已授权的动态对照；不重推理/训练。
检查旧 12 项 rho 与原描述一致、研究 variability rho<=原 rho、C/M/A 嵌套、0/5/10%单调、
未知不默认否决、关闭开关返回原 exposure、日期/路线身份不串接。

## 动态对照（执行前固定，不按结果扩展）

两个条件：q50/M/P70 checkpoint 上的 FROZEN 与 5%研究策略。10:30 相同物理状态出发，
比较 [10:30,10:45) 释放订单；此前等待订单正常继续，但不混入 cohort。
不引入新 dwell、lead、repositioning；30秒滚动、300秒到达耐心沿用 checkpoint。
10:45 后无新请求，继续等待截止并 drain 所有已接受任务。单进程、逐条件冷启动，清理 routing cache。
使用 SINGLE_SOURCE_MATRIX；基线在截断需求生效前与已有 deterministic continuation 对账。
这不是全天结果，也不是原 batch-routing 41 场景的替代。

新 dynamic rho 的语义改变，因此研究分支将 checkpoint 之前实际已选 AV assignment 的
dynamic 累计量按同一 CV/acceleration 定义重新计算，计数、static、speed 及物理任务均不改。
新累计 dynamic 不得大于旧累计值；报告前后值。不能把旧 crawl/stop 累计预算直接混入新量。
已发生的分配不重新规划。新 5% 门控仅适用于 checkpoint 后决策。

输出分配、cohort outcomes、稀疏 epoch 统计、未知暴露、Gamma 状态、资源、来源 SHA。
逐条件1800秒硬超时，RSS 2GiB告警，无GPU/稠密矩阵。失败保存状态、不自动重跑。
完成两个条件后停止；不凭结果挑预算，不自动启动其余 profile 或全天场景。
