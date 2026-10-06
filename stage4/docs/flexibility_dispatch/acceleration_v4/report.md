# Stage4 路由加速 v4：实现与有限预跑结果

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent。
- Origin Mode: 用户授权的代码优化与有限同状态对照。
- Origin Date: 2026-10-06，Asia/Singapore。
- Verification Status: VERIFIED（限定查询/状态）；v4全天未运行。
- Version Label: acceleration_v4。

## 结论

静态基础OD缓存、可切换的单源分组、已知ETA证书已实际接入v4配置及native控制器。
59项相关回归通过，compileall与diff check通过。既有六个状态的有效弧和全部
词典序目标一致；120个waiting批次、每种实现7,200次逻辑查询的接口结果一致。

三轮固定坐标查询：v3 5.588秒 → v4原始缓存2.068秒 → v4原始缓存+分组1.819秒。
最后一个对照约3.07倍，但它包括相同OD的两轮重复查询，**不是全天3倍提速**。
首次查询：1.868 → 1.642秒，约少12.1%；单独加raw缓存的首次查询为1.910秒，
约慢2.3%。缓存并不使一个从未计算过的OD搜索变快，收益来自避免再算。

## 实际实现与不变项

- `routing_v4.py` 在原v3处理顺序外增加最多50,000项raw time/distance LRU。
  完整精度坐标不量化，旧7位别名和其填充先后顺序保留。每次按当前15分钟bin
  重新计算beta，缓存的不是最终ETA。
  原始RAM命中先于可选磁盘查询，不把RAM命中误报为SQLite命中；已有单元测试
  显式禁止该路径调用磁盘读取。本次速度表关闭磁盘，不计这项未经计时的收益。
- 仅允许已检查的Valhalla 3.8.2、matrix时变距离上限0、无实时/远程可变路网的
  静态分支跨分钟复用；配置或引擎不满足时拒绝启用，不静默假设时间无关。
- Loki对整米距离大于0的matrix端点清空date_time；本实现以保守球面距离大于
  10m作缓存技术guard。近零距离保留分钟键并独立1×1，不混入会清除源时间的
  远目标分组。[当前版本源码](https://github.com/valhalla/valhalla/blob/3.8.2/src/loki/matrix_action.cc#L34)
- 分组为相同完整精度源/出发分钟、最多32 targets；失败cell仍退回独立1×1。
  在本有限集合中fallback为0，不将有限一致性推广为任意路网/版本的定理。
- 已知ETA证书使用原patience/session比较及同样的浮点加法顺序；边界相等允许。
  原始查询未知时不以平均速度猜测下界。原Top-K不补位；原有patience/session
  归因及计数保留。prospective funnel开启时不使用此提前证书过滤。
- 保留原v1/v2/v3路径；新入口配置是
  `stage4/config/symmetric_flexibility_acceleration_v4.json`，使用独立输出/磁盘目录。
  仍然两路由进程、2GiB进程组上限、512MiB可选磁盘缓存，不用GPU/稠密矩阵。
- M3、beta输入、C/M/A定义、Top-K20、30秒rolling、300秒patience、600秒时域、
  三Train情景、当前最优面、10秒求解预算均未修改。没有重新训练或重跑全天。

## 六状态预跑

命令：`python -X faulthandler -m stage4.analysis.acceleration_v4_prerun`。
沿用v3的08/10/12/15/18/21时历史可见状态重建，不新造native策略轨迹。
每状态最多20个waiting批次，优先顺序沿用原waiting顺序，不声称随机代表性。
冷查询先预热actor/tiles，再清空OD/ETA缓存，不是冷进程首次加载耗时。
额外+60s/+900s只改变相同坐标查询的时间/beta-bin，不推进物理状态、不改
订单释放链，不作为新动态实验或全天服务率证据。

| 查询段，六状态合计 | v3默认 | v4 raw缓存 | v4 raw缓存+分组 |
|---|---:|---:|---:|
| 原时刻首次查询 | 1.868 s | 1.910 s | 1.642 s |
| 固定坐标跨分钟 | 1.860 s | 0.079 s | 0.093 s |
| 固定坐标跨beta-bin | 1.860 s | 0.079 s | 0.083 s |
| 总查询时间 | 5.588 s | 2.068 s | 1.819 s |
| 实际raw arc计算 | 6,636 | 2,224 | 2,224 |
| 后端matrix调用 | 6,636 | 2,224 | 938 |
| 逻辑arc lookups | 7,200 | 7,200 | 7,200 |

少算约66.5%的raw arcs，少约85.9%的后端调用。逻辑候选没删：仍评估同样7,200次
查询，区别是共享搜索及复用同一基础OD，当前beta和bin结果仍逐次对照。
近零距离等保留时间的12条重算没有被强行合并。
路由时间、距离、corrected ETA、beta、bin逐值一致；既有六状态全部目标在
1e-6内一致。前20个waiting的冷样本只有4条有效弧，不据此估计全日可行率或
剪枝比例；完整模型检查保留其余原始弧，不缩小模型证明性能。

本轮预跑记录：09:31:18–09:32:03（+08:00）；monotonic runtime=45.735秒。
父进程峰值RSS318.52MiB；进程组边界采样最高631.26MiB。后者不是连续峰值。
退出码0、stderr为空；保护输入前后SHA一致，新增native科学条件数0。
每个变体测一次并交替执行顺序，没有置信区间/显著性声称。

## 已知ETA剪枝的真实接口正例

六状态首次查询之前未持有已知OD，所以该段ETA证书命中0，不给它计独立收益。
补充命令：`python -X faulthandler -m stage4.analysis.acceleration_v4_prerun --known-eta-check`。
只复用已检查08:00状态，20批/400逻辑弧。填充原始OD缓存后清空旧的epoch/
最终ETA缓存，仍以同一个时刻/物理状态请求，应用原patience/session预算。

- 399条已知必不可行弧被剔除；原有效弧1条，之后仍1条且ETA相同。
- 热步骤新增后端调用0、raw arc计算0。未知OD未被猜测过滤。
- 检查内部runtime=2.672秒，进程组RSS快照578.93MiB，退出码0、stderr为空。

这证明缓存与证书实际生效，不是仅增加开关；但不能把这些399条再与缓存节省
相加。证书没有证明能独立消除未知OD搜索，也不按这一个偏向旧waiting的状态
推算全日拒绝率。

## QA与停止边界

59项相关回归/compileall/diff check通过。首轮小测试17通过1失败，是新测试将
session终点相等误写成拒绝；修正测试为相等允许，没有修改科学边界。
旧pytest缓存目录权限警告通过关闭cacheprovider避开，未更改该目录权限。
两次有限接口检查均正常结束，不存在native异常/重试或数据/阈值调参。

全天收益仍未测：实际车辆位置、等待队列、50,000项LRU命中和不同策略间重合
决定可省多少搜索。不能承诺107分钟直接除以3.07，也不能说已有30分钟全天结果。
本轮到实现、短对照、提交/推送为止；后续v4全天执行需用户另行指令。

可读机器结果：[六状态结果](prerun.json)、[热缓存ETA检查](known_eta_check.json)。
