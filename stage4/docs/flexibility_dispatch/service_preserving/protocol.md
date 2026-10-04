# 保护当前服务的前瞻：固定两窗口执行协议

## Material Passport

- Origin Skill / Mode：academic-research-suite / experiment-agent / run。
- 日期：2026-10-04；版本：service_preserving_v1；运行前状态：NOT_RUN。
- 用户授权：完成一页既有结果诊断、独立策略及剩余时长处理、M 的两个既有窗口。
- 原批次基线：`57636ec`；不覆盖其18组结果，不改论文，不追加全天实验。

## 1. 决策定义

令 \(c(x),m(x),b(x)\) 为当前 critical、总匹配、carry-over 服务数。
先固定与 MYOPIC 相同的当前 lexicographic 最优面：

\[
\mathcal X_t^*=\operatorname{arglexmax}_{x\in\mathcal X_t}(c(x),m(x),b(x)).
\]

只在这个面内最大化三种 Train 需求情景的期望下一服务数，再最小化接驾时间：

\[
\operatorname{lexmax}_{x\in\mathcal X_t^*}
\left(\sum_s p_s Q_s(x),-\sum_{vo}p_{vo}x_{vo}\right).
\]

新策略名 `SERVICE_PRESERVING_LOOKAHEAD`。没有新权重、风险指数或“AV优先”叠加。
每轮另用既有稀疏 MYOPIC 求解器核对前三层计数，记录最优值与实际值；这是必要数学核对，不是新性能 Gate。
保持当前计数不意味着服务对象相同，也不保证整个窗口的服务量优于 MYOPIC。

## 2. Train-only 条件剩余时长

现有 \(\widehat T_i=\sum_l\widehat\tau_{il,50}\) 是 traversal 中位时长求和；不能声称为整单 joint P50。
replay 的 \(T_i\) 为 Stage1 arrival−departure 的 GPS 观察窗时长，未识别真实车门事件。
用既有 Train 模板（20161010/17/24）及同订单 Stage1 观察窗构造 \(R_i=T_i/\widehat T_i\)。
只使用正值、有限且可对账的 pair；不裁剪、不分箱、不搜索修正倍率。

对已接受、仍 observed busy 的任务，令 \(a\) 为已过的载客预测起点时间：

\[
\widehat T^{rem}(a,\widehat T)=
\widehat T\operatorname{median}\{R_i:R_i>a/\widehat T\}-a.
\]

服务未开始时先等待已接受任务的接驾/overhead结束，再使用 \(a=0\) 的估计。
仅由可观测 busy、已接受任务时间与预测、Train empirical distribution 计算；不读取实际 service_end 或未接受任务的未来终点。
经验尾部无支持时，省略这个 busy 的 prospective state，实际任务照常执行，不取消、不提前放车。
current admission、当前候选和新任务 M3 时长均不修改；修正仅用于已接受 busy 任务的未来可用状态。
该适配器校准 replay 代理口径，不是 Stage2 重训或真实机器人出租车时长认证。

## 3. 唯一新增的两个条件

- Test31，M，10:30 与17:30的原共同物理 checkpoint。
- 各纳入随后15min全部请求，30s rolling、300s patience，完成 draining。
- MYOPIC、AV_FIRST和旧 LOOKAHEAD 六份 reference 全部复用，不重跑。
- 基础配置、C/M/A规则、5%环境外预算、Top-K20、future K5、2km、需求强度×3、forecast seed均不改变。
- Gamma/cost/repositioning仍关闭；单进程 CPU，无 GPU、无稠密订单×车辆矩阵。
- 20,000变量/150,000非零元、单轮10s、单条件1,800s、RSS提醒2,048MiB继承原协议。

目标函数和条件时长同时变化，不能把新旧差异分别归因于其中一个，不再追加 factorial ablation。
报告配对 gained/lost、服务数、AV分工、等待、当前最优面保护和资源；不将 recourse 当成已实现服务。
若仍无清晰收益，结束此批次，不继续调参；保留 MYOPIC 主参照。即使改善，也只称两个开发窗口内的结果。

## 4. 执行命令

```powershell
python -m stage4.analysis.flexibility_diagnostics
python -m stage4.analysis.flexibility_service_preserving --fleetpy-root D:/pycodes/didi_xian_raw/.external/FleetPy
```

实现/协议在运行前提交；结果另提交。代码、冻结输入和模型读取来源保存在简明summary，不新建多层evidence体系。
