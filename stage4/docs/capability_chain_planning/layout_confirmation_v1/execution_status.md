# 六组执行状态：6/6完整完成

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent（执行监测）。
- Origin Date: 2026-10-13（Asia/Singapore）。
- Verification Status: SIX_COMPLETED_EXECUTIONS_ANALYZED。
- 实际生产代码：`c712e78cdd1ebea8df251988ff71b9308dce300c`；代码及固定六组配置未改。

## 最终状态

原定六组全部完整完成、退出0，三对聚合与最终汇总均通过，全部实验进程已退出。没有新增条件、改代码／参数／时限或重新生产上游。完整数据及环境事件见[最终报告](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/layout_confirmation_v1/full_report.md)。以下为原始启动事件的历史记录。

新10%种子条件的两份布局已完成：48批、每份838个C标签窗口，431个位置相同、409个不同。布局处理61.6247秒，峰值244.1211 MiB；进程正常退出0。布局SHA256为`0408d96081c17b6bee6ea9dd26781f00be55c179731e2e3ad6ae3053d1e4d9ce`。

首次第1组C兼容热点全天子进程于2026-10-13 13:48:14（UTC+08）启动，约3.6327秒后退出1。`last_tick_s=null`，只有启动失败摘要，没有任何仿真步骤、订单服务结果或路由产物。

用户已明确授权修复这类启动环境问题，并要求以后无需按技能流程反复请求确认。失败启动目录已原样移到`Q10_LABEL_SEED2/failed_startups/native_001`，可恢复，布局未重新计算。为当前PowerShell进程追加一个仅指向固定FleetPy目录的`safe.directory`环境配置，保留原有Git配置项；该设置由仿真子进程继承，未写全局配置。

第1组于2026-10-13 15:16:42（UTC+08）原样重启并最终完整完成27666单。新10%联合随后发生一次磁盘环境中断，保留部分现场后原条件重启，仅复用完整第1组；最终新10%联合完成27671单。5%两组完成27667／27678单，20%两组完成27551／27536单。全部均完成排空与物理对账。

## 原因与已核对证据

FleetPy读取版本号时调用`git -C <FleetPy> rev-parse HEAD`，被Git目录所有权检查拒绝：目录属于本机用户，执行环境为沙箱用户。这是启动环境问题，不是模型、求解器或服务率结果。

使用仅作用于只读诊断命令的目录信任声明后，先核对固定FleetPy版本；随后经用户授权，在实验进程中设置相同的明确目录信任并重启。没有写全局Git配置。首次启动时可用内存4479.23 MiB、磁盘10391.82 MiB，没有资源不足；其后第二组的短时磁盘中断是另一件事，详情及完整归档在最终报告。六组配置与布局摘要始终保持不变。

本地原始错误及退出回执分别位于：

- [保留的启动错误尾日志](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/output/capability_chain_planning/layout_confirmation_v1/Q10_LABEL_SEED2/failed_startups/native_001/logs/COMPATIBLE_HOT_COMMON_CONTROL_tail.log)
- [保留的失败退出回执](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/output/capability_chain_planning/layout_confirmation_v1/Q10_LABEL_SEED2/failed_startups/native_001/execution_receipts/COMPATIBLE_HOT_COMMON_CONTROL.json)
- [已完成布局摘要](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/layout_confirmation_v1/Q10_LABEL_SEED2/layout/preparation_summary.json)

## 收尾边界

对已明确定位、不会改变科学条件的启动环境问题，按用户新约定直接修复并保留现场。对模型、物理对账、资源上限或正式仿真中的失败，仍须独立判断，不据此自动修改策略/阈值/种子/时限。

只交付完整六组结果、只读汇总与图，不将任何部分数据当全天结果。研究比较已完成，不追加实验；结果与用户对环境故障的处理约定一起提交推送，仿真生产SHA仍明确记录为`c712e78`。
