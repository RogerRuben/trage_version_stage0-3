# V2首次执行与空间邻域边界修复

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent，run。
- 日期：2026-10-15，Asia/Singapore。
- 执行代码：`442f8106a33ec966c891c9288cc05a7ceffa7b6e`。
- 执行状态：**CRASHED_EARLY_CODE_ERROR**；修复状态：**FOCUSED_CHECKS_PASS_NOT_RERUN**。
- Verification Status: UNVERIFIED（城市速度与服务效果尚未验证）。
- 本说明仅记录执行和确定的代码修复，不添加实验组、阈值搜索或额外放行程序。

## 执行结果

在工作目录`D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla`执行：

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:CUDA_VISIBLE_DEVICES='-1'
& 'D:/anaconda/envs/stage0-valhalla/python.exe' -B -u -m stage4.analysis.bottleneck_whole_method run --execute --config stage4/config/bottleneck_whole_method_v2.json
```

启动前未发现遗留Python仿真进程；可用RAM约5376 MiB、磁盘约11597 MiB。协调器17212仅启动空间联合子进程28936；二者随后均已退出，未启动瓶颈联合组。原独立参考组和47批布局没有重跑。

- 子进程退出码：1；启动至退出73.55秒，未触发3小时行政时限。
- 原生runner记录运行72.43秒，其中初始化71.22秒。
- 最后完整仿真步180秒（00:03），共7步；该前缀循环耗时0.553秒。
- 失败发生在210秒（00:03:30）的联合空间邻域构建处，`KeyError(115)`。
- 已接单、完成单、过期单均为0。没有有效的全天服务结果，不能由前7个安静步推断全天速度。
- 最后完整步RSS约830 MiB、系统可用RAM约4687 MiB；本次不是内存上限、显存或稠密矩阵问题。

本地失败现场：[summary](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/output/capability_chain_planning/bottleneck_whole_method_v2/native/SPATIAL_SHARED_PLAN/summary.json)、[子进程收据](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/output/capability_chain_planning/bottleneck_whole_method_v2/native/receipts/SPATIAL_SHARED_PLAN.json)、[错误日志](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/output/capability_chain_planning/bottleneck_whole_method_v2/native/logs/SPATIAL_SHARED_PLAN.log)。部分Parquet仅用于诊断，不能精确恢复仿真物理状态。

## 确定的错误及最小修复

错误只在以下组合出现：最近六辆都是当前动作固定的资源；种子订单的现有拥有者不在这六辆内；还需加入一辆能改变当前动作的资源。

1. 先加入外部拥有者时，删除原最近邻列表的最远成员。
2. 再加入可改变动作的资源时，旧代码再次按原列表删除同一最远成员。
3. 成员已不存在，集合`remove`抛出异常。这不是求解器难解或搜索达到预算。

现在第二次替换只从**当前仍存在**的邻近成员中选最远者；容量允许时保留订单拥有者。单车邻域无法同时容纳拥有者与活动资源时，仍保留外部容量约束，不绕过共享订单保护。没有吞掉异常、放松六车限制或扩大搜索域。

修改只涉及空间参考的邻域成员选择，瓶颈闭合、共同动作、能力/乘客规则、共享请求容量、初始化、历史情景、路由与物理执行均未修改；300秒接单、600秒接驾和每轮10秒预算不变。

## 必要检查与停止点

新增一个参数化回归用例，覆盖六车实际触发组合与单车边界，验证成员不重复删除、当前可变资源被纳入、容量不超限，并在六车情况下保留现有拥有者。

原四份相关测试加该回归合计**24项通过，2.58秒**；相关文件compileall及diff check通过。这些是构造检查，不是城市性能基准。

没有重跑失败组或启动第三组。已向用户请求修复后原条件继续两组的确认；现有失败产物和旧参考/布局全部保留。后续若获准继续，应先将这次失败目录完整移入明确的attempt归档，不能覆盖或把它标成完成；两组新运行必须绑定同一修复提交。

当前不能给出V2提速倍数、全天耗时估计或算法服务排名。只有完成实际运行后，才可比较相同仿真前缀的计时，并据分段计时确认优化是否生效。

## 后续授权运行与用户停止

用户随后明确允许修复后原条件继续两组。首次失败的`native/`及当时`setup/`完整归档至`stage4/output/capability_chain_planning/bottleneck_whole_method_v2/attempts/20261015_early_keyerror_442f810/`，未删除；修复提交`fba25dc`从新目录、午夜开始运行空间联合。原参考与布局均未重跑。

该运行越过首次错误位置，但实际计时未见整体提速。用户再次明确选择“停止两组队列，先分析搜索开销”。2026-10-15 19:53:03（+08:00）已停止协调器20996及子进程15692，确认无残留比较进程；最后持久化到00:40、81步，瓶颈主方法未启动。本页开头的未重跑状态是首次失败后的历史停止点，不是最新队列状态。

最新结论和数值见[V2搜索开销诊断](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/bottleneck_whole_method_v2/search_cost_diagnosis.md)。当前只有部分运行诊断，没有全天服务排名或提速优势，也没有自动重启计划。
