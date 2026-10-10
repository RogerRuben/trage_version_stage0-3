# Stage4 算法与工程加速 v1

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent。
- Origin Date: 2026-10-05，Asia/Singapore。
- Scope: 代码优化、必要数学对照、有限微基准；不是新native科学场景。
- Status: 小范围功能/性能已实测；全天加速比未验证。
- 30分钟是参考目标，不是新Gate；不调质量、兼容阈值、情景数、600秒horizon或30秒rolling。

## 1. 已实现的独立加速路径

基础配置`stage4/config/symmetric_flexibility_full_day_v1.json`保持原样。加速配置只允许技术参数，输出写到`stage4/output/symmetric_flexibility_v1/accelerated_full_day/`，不覆盖原结果。

1. 当前状态选择x保持二进制；未来匹配y连续化，并恢复真正的整数匹配。
2. 每情景、每状态的总容量门控取代独立edge门控。对整数x等价，LP界更强、约束更少。
3. 预索引Train释放时间、预计算位置字符串和兼容集合；保持Poisson、抽样、jitter及接受随机数的原调用顺序。
4. 路由保持相同1x1 primitive，2个常驻worker仅处理独立查询。结果全部返回、按原输入顺序读取，再派单。
5. legacy单轮缓存照常清理；独立有界LRU只复用完整精度位置、路由模式、分钟、bin和beta完全相同的成功查询，最多50,000条。
6. 保留可选持久化HiGHS后端，在同epoch目标层之间更新模型并传递初解；不跨时刻盲目复用forecast IDs、解或cuts。

默认路径仍为BINARY、SCIPY、原抽样和串行路由。新加速配置默认FLOW_RELAXED、SCIPY、预索引、LRU和2个worker；可选Highs不默认启用，因为当前微测没有额外收益。

## 2. 数学等价范围

固定整数状态选择x后，每辆车只剩一个有效下一服务起点。删去当前已服务订单，各情景只剩每车、每订单容量为1的二部匹配。时间、会话、车型及乘客约束已体现为允许边。

因此每个情景的连续匹配LP存在整数最优解，Q_s^LP(x)=Q_s^IP(x)。保持x整数后，各层当前优先级、未来期望服务及ETA目标的最优值保持不变。[二部匹配整数性](https://math.mit.edu/~goemans/18433S15/matching-notes.pdf)

总状态门控sum(y_state)<=x_state在x为0时禁用该状态，在x为1时不比原每车容量更严格；它与原整数问题等价，不是删候选。

实现以稀疏最大匹配恢复future records，并检查加权未来计数与LP目标一致，不能对y按0.5取整。没有全域稠密矩阵。

这个论证不适用于额外的跨车全局recourse资源预算、多次串行未来任务等扩展。它也不意味着整个含x的模型是LP整数多面体。并列最优选择、数值容差和限时回退可能改变全天轨迹；不声明bitwise复现旧执行。

## 3. 有限性能实测

完整数据见[benchmark_summary.json](benchmark_summary.json)。均为单次、固定顺序微测，不是全天速度或统计显著性结论。

| 项目 | 原路径 | 加速路径 | 观察 |
|---|---:|---:|---|
| 合成小稀疏问题，465变量 | 0.330秒 | 连续recourse SciPy 0.045秒 | 约7.3倍；整数变量465→38 |
| 合成中稀疏问题，1,228变量 | 0.741秒 | 连续recourse SciPy 0.144秒 | 约5.1倍；整数变量1,228→119 |
| 同一两实例，可选持久化HiGHS | — | 0.050 / 0.158秒 | 略慢于连续recourse SciPy，保留可选 |
| Train抽样8窗口、4,988请求 | 0.512秒 | 0.050秒 | 约10.3倍；完整Scenario对象相等 |
| 100批、2,000路由arc lookups | 3.327秒 | 2 worker 2.104秒 | 含worker首次启动；ETA/距离/bin/beta逐项相等 |
| 同批相同输入，30秒后复用 | — | 0.039秒 | 仅证明缓存复用，不代表全天命中率 |

预索引初始化0.629秒，原路径0.411秒：增加约0.218秒一次性成本。路由测试进程组RSS快照394.1 MiB，父进程峰值235.8 MiB，无GPU。完整native的内存会更大，已设置进程组2,048 MiB上限；不将此快照当全天峰值。

各模型五层目标向量一致，恢复future匹配不重复车辆/订单、不重复计入当前已服务订单。专项测试28项通过（1.49秒），覆盖critical/carry优先级、非均匀情景概率、预测边界与缓存；compileall、CLI help、diff check通过。技术微测不解释策略服务效果，也不能把各项加速比相乘。顺序执行存在OS缓存和CPU状态差异。

## 4. 复现与执行边界

必要测试：`python -m pytest -p no:cacheprovider stage4/tests/test_flexibility_model.py stage4/tests/test_flexibility_native.py stage4/tests/test_routing_determinism.py stage4/tests/test_symmetric_research.py stage4/tests/test_acceleration.py -q`。

微基准：`python -m stage4.analysis.acceleration_benchmark --with-routing`。脚本拒绝在同目录native进程存活时执行真实并行路由测量。

可选HiGHS固定版本1.12.0只安装到`stage4/output/runtime_dependencies/highspy_1_12_0`，不更改conda、NumPy或SciPy。默认SCIPY路径不需要它。[HiGHS接口](https://ergo-code.github.io/HiGHS/dev/interfaces/python/example-py/)

加速执行入口（后续授权见下节）：

```powershell
python -m stage4.analysis.symmetric_flexibility_full_day --fleetpy-root D:/pycodes/didi_xian_raw/.external/FleetPy --policy SERVICE_PRESERVING_LOOKAHEAD --acceleration-config stage4/config/symmetric_flexibility_acceleration_v1.json --administrative-timeout-s 21600
```

此前用户选择“先完成代码优化，不重跑全天”，该阶段仅完成有限QA与提交。后续运行授权单独记录如下。不新增C/A、日期、接受率或配额条件；10秒求解、20,000变量/150,000非零元上限不变。

## 2026-10-05 后续检查与新授权

用户随后明确要求“接着完成刚才升级代码后的检查与分析，思考还能不能优化之后运行实验”，覆盖前述本轮暂不运行的指令。仅执行相同M条件的第二组，复用完整MYOPIC，不新增C/A/日期/参数条件。

追加实现`lock_current_face`：在当前小图上求critical/current/carry最优值，固定相同最优面，仅在扩展模型求future和ETA。y=0对任意可行当前分配可行，所以前三层不受未来情景影响。默认仍关闭，独立可切换。

实测FLOW_RELAXED SciPy无锁/有锁分别0.0438/0.0430秒和0.1438/0.1475秒，目标向量相等；总耗时没有稳定收益，故本次正式配置关闭该开关。Highs仍非默认。不同结果仅根据技术计时选择，未依据服务结果调参。

新入口支持从旧目录复用已完成MYOPIC，并核对双方共同冻结输入SHA一致。旧中止产物不覆盖。新隐藏后台runner记录stdout、stderr、PID和退出码，启用Python faulthandler；不依赖旧的临时exec会话恢复退出原因。

## 5. 原任务现状与QA异常

MYOPIC完整结果保留，不重跑。原第二组6小时尝试的进程目前不存在，最后保存于2026-10-05 00:44、模拟14:30；原始summary仍为RUNNING，无法读取原exec退出记录，不猜测退出原因，不当成完成结果。此前优化阶段按用户答复没有重跑；本次按照上节新授权，只在独立accelerated_full_day目录运行第二组。

首轮微基准计算结束后因新文档目录不存在而写出失败；已修复显式创建目录，再运行约9.05秒成功落盘。它不是native处理异常或科学结果重试。

## 6. 本次授权执行已经完成

2026-10-05在独立目录完成第二组，执行代码6b46f54，退出码0；MYOPIC未重跑。服务18,702→19,398，平均已服务等待170.774→181.427秒。运行135.980分钟，进程组step采样峰值1,024.496 MiB，回退/当前最优面违规/路由失败均0。实际前瞻流水线64.535分钟、路由52.435分钟，不再把纯MYOPIC的“路由几乎独占瓶颈”直接套到两阶段策略。详细结果与后续优先级见[完整执行报告](report.md)。
