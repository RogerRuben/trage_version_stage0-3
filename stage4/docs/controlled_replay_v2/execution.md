# Controlled Replay v2：首轮两组执行记录

## 范围与预定协议

2026-10-06作者明确授权“开始实验”。本轮执行此前确认的[独立环境设计](design.md)，不覆盖旧冻结模拟；不增加日期、车型、车时比例或参数搜索。

两组均为20161031、同一批原始30,000订单及共同输入支持、同一8435个有效班次模板、同一初始位置、同一30秒滚动前瞻派单算法。上车时间作为请求释放时间继续作为已接受的研究近似，接驾耐心300秒。M3、Stage3研究兼容性、5%环境外暴露预算和10秒单轮求解上限不重选。

| 条件 | 预定AV车时目标 | AV乘客接受率 | 车辆研究能力 |
|---|---:|---:|---|
| PURE_HV | 0% | 100%（无AV，参数不生效） | HV |
| M_Q10_P70 | 10% | 70% | HV + M |

逐时**总预定供给严格相同**。混合组按预先确定的分层哈希前缀得到实际AV日内车时10.00048%，不按服务结果调整标签。小时AV占比可以不同于日均目标，24小时全部报告。班次窗口是停止接新单的边界，已承诺任务允许跨窗口完成；实际窗口后工作时间另外记账。

两类车共同采用历史09–24订单的15分钟起点分布进行短程空闲调位：正常派单优先、2km内最近3个有缺额目标、校准空驶时间不超过300秒且预计不越班次，每轮全车队最多50次。仅日内96个边界启动。没有客户承诺的空车移动可以被真正接单中断，保留实际已行驶前缀。位置沿scalar route形状推进，不直接沿起终点弦线移动。坐标到路由吸附端点的连接显式保存、报告，不宣称连接是单独匹配的道路。

新空驶路径的路网身份、转向、信号与已认证禁止动作使用现有Stage3解析接口。新路径没有精确冻结M3输入行，交通证据明确标U，不复制邻近订单预测，也不填入真实未来拥堵；沿用已冻结研究规则对U的处理。本轮不把空驶宣称为完整动态ODD评估或AV安全认证。

## 最小工程检查

- 编译检查通过；共同供给、班次接单、前瞻机会、路由缓存、物理移动等相关回归 **54项通过**（含已有测试，不是新建54项）。
- 同窗口、同稀疏当前图、乘客接受且无额外成本约束时的HV/A标签互换，当前选择完全一致；两个类型的native接单窗口边界均已检查。这是必要小样本一致性检查，不是另一次全天A条件。
- 单条真实scalar route + edge_walk + production parser连通，发现真实掉头/环岛后C/M拒绝、A/HV一致；没有修改规则以通过检查。
- M_Q10_P70独立一小时工程运行正常退出：原始356单，共同341单，服务310单，31单耐心到期，15单共同输入不合格；51次空驶中33次完成、18次因接客中断，物理记账通过。
- 一小时实测总耗时37.5秒，峰值进程组RSS821.375 MiB，private committed峰值2094.75 MiB（与RSS不是同一指标）；GPU未使用、没有稠密矩阵。空驶候选105条中16条共同路线证据未通过，只放弃该调位目标，两种车型同等处理。

工程窗口在提交前运行，summary的HEAD为565995d，实际包含本轮未提交实现；因此不把其HEAD误称为完整代码绑定。**正式两组全天在实现提交完成后串行执行，summary记录实际execution_code_sha。** 只报告已落盘的全天结果，不按一小时窗口外推性能或成功率。

## 执行与资源

Python环境：stage0-valhalla；固定FleetPy位于 `D:/pycodes/didi_xian_raw/.external/FleetPy`。CPU数学线程各1、路由worker 2、CUDA关闭。进程组RSS上限2048 MiB；private committed另报。路由可选磁盘缓存512 MiB，空驶缓存1024条；不构造订单×车辆或全路网点对矩阵。单轮求解10秒，单组行政上限10800秒，物理任务排空上限为原有172800模拟秒；行政上限不改变客户任务语义。

```powershell
python -u -m stage4.analysis.symmetric_flexibility_full_day `
  --fleetpy-root D:/pycodes/didi_xian_raw/.external/FleetPy `
  --policy SERVICE_PRESERVING_LOOKAHEAD `
  --acceleration-config stage4/config/symmetric_flexibility_acceleration_v4.json `
  --controlled-scenario PURE_HV

# 第一组正常退出后，才启动第二组。
python -u -m stage4.analysis.symmetric_flexibility_full_day `
  --fleetpy-root D:/pycodes/didi_xian_raw/.external/FleetPy `
  --policy SERVICE_PRESERVING_LOOKAHEAD `
  --acceleration-config stage4/config/symmetric_flexibility_acceleration_v4.json `
  --controlled-scenario M_Q10_P70

python -m stage4.analysis.controlled_replay_results --root .
```

完整本地路线与订单输出在 `stage4/output/controlled_replay_v2/runs/<condition>/SERVICE_PRESERVING_LOOKAHEAD`；短窗口在单独smoke目录。结果解释见完成后生成的[两组报告](results/report.md)、[比较摘要](results/comparison.json)及[逐时供给表](results/hourly_supply.csv)。旧82.40%/68.58%只作为旧组合环境背景，不用来替代新环境配对结果。本轮两组改变了类型及其乘客/能力适配，不能仅凭差值分离每一类约束的独立效应，也不能从单一日期声称普适显著性。

输入来源由每组summary平面记录SHA；没有新建多层evidence体系。若出现处理错误、资源超限或时间超限，保留局部文件并报告，不调整科学阈值来凑完整结果。

本记录为AI辅助实现与实际运行整理；用户授权及科学定义保持独立于实验结果。

## 已完成结果（2026-10-07汇总）

两组均完整结束，进程退出码0，执行版本均为`c1e6851e351a0b4dce1e9c1bc7891338cb36f5b5`。纯HV服务24,810/28,367（87.4608%），M混合服务24,632/28,367（86.8333%）；差值为混合少178单、共同口径低0.6275个百分点。原始30,000单口径分别82.7000%和82.1067%。

两组实际耗时分别31.14895和30.71902分钟，峰值采样进程组RSS分别1350.48和1338.52 MiB，无GPU、无稠密矩阵、无求解资源回退。全部24小时总预定供给一致，15项共享输入一致且未变，客户任务/接单窗口/空驶前缀记账对账未发现违规。仅生成本轮两组，不追加新的模拟。

最终逐项账、条件性配对等待、空驶和耗时拆分见[完整报告](results/report.md)。新环境两组差距较小，但单日、单种子与组合条件的比较不等于某一门槛的独立因果效应，也不据此覆盖旧冻结结果。
