# Verification summary

日期：2026-09-14。开始基线 `5671c3aca81a9ddd2616297906617c78764fb533`；分支 `codex/stage2-v5-micro-transfer`。只新增本目录文档与一个独立合成核验脚本，不改生产代码或配置。

## 实际执行

已核对任务书所列本地系统材料、外部定位材料；已针对四篇 closest works 核对可访问的出版社或作者稿信息。访问限制和未核准内容记录在 [文献修正](closest_work_corrections.md)，不声称全部文献全文已核验。

使用现有环境运行：

```powershell
& D:/anaconda/envs/stage0-valhalla/python.exe stage4/tools/dynamic_control_toy_check.py
```

单进程、CPU、Python 标准库，仅 Fraction 与有限子集枚举。执行返回 `status=PASS`、`declared_cases=9`、退出码 0；PowerShell Stopwatch 测得本次进程调用约 0.141 秒。没有使用 GPU、网络路由、真实订单、生产导入或矩阵分配。没有测量峰值 RSS，不填造内存指标。脚本只向 stdout 输出小 JSON，不保存实验产物。

执行前限定的九项为：四个动作×情景图，以及五个概率值。每项内部的公式断言不扩展成新实验网格。

| 核验对象 | 结果 |
| --- | --- |
| x_A/W1、x_A/W2、x_H/W1、x_H/W2 可行弧与匹配 | 2、1、1、2，与解析一致 |
| p=0、1/4、1/2、3/4、1 | V_A=1+p、V_H=2−p，精确分数一致 |
| 同信息简单阈值规则 | 五个 p 均等于非预知最优 |
| 信息放松 | 各 p 精确值 2；与最优差 min(p,1−p) |
| 固定随机混合 q=0、1/2、1 | 同五个 p 内核查 V_q 公式且不超过最优 |

## 纸面核查

- t=600 释放、t=900 未来到达、t=960 deadline；跨区 600 秒不能接驾，一车不能先后完成两张未来单。
- 当前订单计数只加一次，未来值不含当前单；事件概率不双乘。
- Route A 保留 critical→total→carry-over；strict 固定最小 pickup，bounded 仅符号化 aggregate 秒预算。
- 全车队机会成本没有丢掉 A；session 边界不被误写为必然翻转。
- 需求资格与车辆 pickup/session 弧区分；不把聚合供给视为已证明充分状态。
- p 是给定分布而非未来实现；非预知最优与有限 clairvoyant 上界分开。

上述 Route A 与信息/时钟结论是定义和解析核对，不是生产 solver 行为测试。完整分类见 [README](README.md)。

交付前逐一解析五份 Markdown 的本地链接，目标全部存在（LOCAL_LINKS_PASS）。Git 范围检查只发现本轮五份文档和一个脚本，没有已跟踪生产文件改动；提交前另检查暂存差异的空白错误。

## 明确未执行

没有生产测试套件、compileall 全仓扫描、真实数据试跑、训练、M3 推理、Valhalla、FleetPy、全天实验、额外机制实验或性能 benchmark；没有修改 V3、41 个冻结场景、上游产品和阈值。没有安装依赖、复制大数据或构建 evidence bundle。

本轮支持有限模型自洽与公式正确性，不支持真实数据收益、预测可用性或复杂方法必要性。交付后停止。
