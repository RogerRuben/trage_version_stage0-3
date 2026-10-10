# 路由加速 v4：本轮范围与有限对照

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent。
- Origin Mode: 用户授权代码优化与有限同状态预跑。
- Origin Date: 2026-10-06，Asia/Singapore。
- Verification Status: UNVERIFIED（执行前计划）。
- Version Label: acceleration_v4。

## 授权范围

只落实静态原始 OD 缓存、最多 32 targets 的单源分组查询、已知 ETA 证书剪枝。
不启动 native 全天、不新增日期/profile/策略，不改 Top-K20、30秒 rolling、
300秒 patience、600秒 horizon、三 Train 情景或10秒求解上限。
v1/v2/v3配置及结果保留，v4使用独立配置与目录。两路由进程、2GiB进程组提醒
上限、50,000项原始OD内存缓存、512MiB可选磁盘缓存；不用GPU或稠密矩阵。

## 实现原则

1. 仅在冻结 Valhalla 3.8.2、matrix时变距离上限为0、无可变远程/实时数据的
   已确认静态分支复用跨分钟 raw time/distance。近零距离和其他模式保留时间键。
   beta及其15分钟bin每次按当前时刻重新计算，最终ETA不跨bin直接复用。
2. 保留既有7位坐标缓存别名的先后顺序，不将完整精度坐标再量化。
3. 已知缓存ETA若必然超过原patience/session边界，可在进入候选图前剔除。
   原始查询未知时不使用平均车速猜测下界。此证书与原始缓存合用，不宣称
   对未知OD有独立的查询削减收益。Top-K不补位，原有拒绝归因及计数保留。
4. 分组仍为前向单源、最多32目标；失败cell回到原1×1。有限对照结果不推广为
   任意路网/版本的普遍等价性。

## 有限执行

命令：`python -X faulthandler -m stage4.analysis.acceleration_v4_prerun`。
工作目录：当前stage0-v6-valhalla工作树；既有stage0-valhalla环境；线程1。
沿用08/10/12/15/18/21时六个历史可见状态，最多20个waiting批次/状态。
比较v3原路径、v4原始缓存、v4原始缓存+分组的相同OD查询；另在固定坐标下
改变分钟及beta-bin测试接口重用，不称为新的动态仿真实验。
对真实状态比较有效弧及各层目标，不根据服务结果调参。
总行政上限300秒；监控进程存活、进度和RSS，进度写入独立目录。

输出：本目录`prerun.json`与`report.md`；临时运行日志在
`stage4/output/runtime_acceleration_v4/prerun/`。只报告实测范围内的收益，不把
重复固定OD的热缓存倍率当作全天倍率。完成必要测试、compileall与diff check后
提交并推送已授权origin，到此停止。
