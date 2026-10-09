# 第四次执行更正：名义30秒尚未真正落实

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent；Origin Mode：run。
- Origin Date：2026-10-09；Verification Status：UNVERIFIED；Version Label：attempt_004。

执行代码 `40b82ab037ff1d2696ef36dd55d2394948561d85`，命令 `python -m stage4.analysis.scheme_a_city_experiment --retry-layout --resume`。进程30016于23:11:28（+08:00）启动，恢复38批／683槽位，运行208.829秒后退出1，没有新增完成批次或启动全天组。

错误仍在第39批19:30的 `expected_empty_distance_m`。诊断记录了请求30秒，但进一步读底层代码发现 `_Budget` 仍执行 `min(seconds, DECISION_TIME_LIMIT_S)`，而常量为10秒。**因此本次实际总预算仍为10秒，不能作为“30秒不足”的证据。** 前一步接口测试没有覆盖这层硬上限，属于代理的实施遗漏，现已明确更正。

本批构图CPU8.935秒、连接证据184.800秒。没有求解器Gap记录，不能推断实际可行解与最优差距。此前38批有效前缀未改变，原现场归档于当前输出目录的 `failed_attempts/layout_001/`；恢复入口已从该原检查点核对并补存，保留原配置与原代码来源。

修正只将显式 `offline_layout=True` 的底层总预算上限设为30秒；实时与默认调用仍硬限10秒，离线也不能超过已经授权的30秒。新增直接覆盖预算类及真实主函数传递的聚焦检查，而不是再次只模拟上层接口。20项检查通过（1.18秒），compileall通过。

同时在恢复后、计算下一批前就写出恢复检查点；失败记录增加实际总预算、该目标剩余时限、求解器状态、返回可行解／界／Gap等少量标量，不导出身份或决策向量，不改变接受判据。

用户已经明确授权真实离线30秒；此项授权尚未被实际执行。修正后沿用该授权完成一次真实30秒恢复，所有科学条件、实时10秒和最优性要求不变，不将上限加到更高。
