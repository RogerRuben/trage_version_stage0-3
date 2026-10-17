# 经济异质联合派单：固定原生短窗检查

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent run
- Origin Date: 2026-10-17
- Verification Status: UNVERIFIED（执行准备完成，尚未运行）
- Version Label: economic_joint_native_check_v1

## 固定范围

只执行Test31的08:15—08:30一个冷启动窗口，随后完成已经承诺的订单和移动。它不是恢复08:15以前的车辆状态，不是全天策略比较，也不用来重选经济参数。

复用47批既有布局、851个C班次槽位、q=0.1原供给、70%乘客接受、固定C能力、M3及原始路线。保持300秒接单期限、600秒接驾上限、30秒决策、10秒决策预算和900秒总时限。新经济目标与上一提交完全一致。

本轮只运行SUBSTITUTION控制器，MYOPIC/SPATIAL和全天比较均不启动。两类连接共用小查询预算；不构建稠密矩阵，不使用GPU。

## 命令和产品

工作目录为本工作树根目录，使用现有stage0-valhalla环境：

```powershell
python -B -u -m stage4.analysis.economic_joint_native_check run --execute
python -B -m stage4.analysis.economic_joint_native_check analyze
```

配置：[economic_joint_native_check_v1.json](../../../config/economic_joint_native_check_v1.json)。原生产品仅写入`stage4/output/capability_chain_planning/economic_joint_v1/native_check/`，已有基线不覆盖。完成后核对每单金额、实际调位距离、服务保障、真实耗时和内存。

增加的轨迹字段只记录即时金额、联合选中金额及紧急/结转服务数，不改变任何决策。原生派单产品也明确标注经济控制器，避免与旧数量目标混淆。

## 运行前检查

27项相关组件/接口回归通过；变更文件编译通过。原生短窗配置成功加载。启动前可用内存约4.8GiB，磁盘余量约11.2GiB；无其他项目Python实验进程。保留1536MiB进程上限和启动内存门槛，不关闭用户其他应用。

## 执行结果

待本轮固定运行完成后填入；此时不报告服务率、C利用率或算法改进幅度。
