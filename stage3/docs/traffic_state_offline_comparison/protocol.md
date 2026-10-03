# 交通状态替代：Validation 离线对照协议

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: authorized descriptive analysis / plan
- Origin Date: 2026-09-27
- Verification Status: UNVERIFIED before execution
- Version Label: traffic_state_offline_comparison_v1

## 研究问题与边界

固定路线、冻结 M3 和 profile，用交通状态暴露替代 crawl/stop 六个 E/Q/C 维度时，
动态部分的判定变化有多少由已知暴露决定、有多少取决于未知状态处理？
不预设新方案更好；不训练、不调生产阈值、不读 Test31、不跑 Stage4。
原方向、静态、速度域、乘客接受和派单逻辑不改。本批不重新评估这些非动态条件。

样本为已有 20161025–27 的 29,851 条 complete dynamic ORIGINAL routes。
这是选中过的路线集合，不是全部路网/全部订单，也不是新的未见测试集。
本批是描述性穷举，不做功效分析、p 值、因果或安全判断。

## 对照定义（执行前固定）

基线：冻结 crawl/stop + speed_cv/acceleration_rms 共 12 项均不超 cap。
替代表达：移除 crawl/stop 六项，保留 speed_cv/acceleration_rms 六项，
以目标环境外已知预测时间占比 U 描述交通环境：

- C: U = congested + severe；
- M: U = severe；
- A: U = 0，仅因草案没有更高的已知类别，不代表 A 全部可行。

Z = unknown + mixed，两者仍分别报告，不从分母删除。
暴露区间 [U, min(1,U+Z)] 是未分类暴露全部在目标环境内/外的保守界，
不是统计置信区间、概率或对 missing 状态的填补。对 A 也保留 Z：
无法确认未知是否属于已建模的三类环境。

对每个公共预算 b=0,0.01,...,1（仅画诊断曲线，不选最优点）：

1. 保留的 variability 家族超限 → RETAINED_VARIABILITY_EXCEEDS；
2. U>b → KNOWN_EXPOSURE_EXCEEDS；
3. U+Z<=b → WITHIN_BOUND；
4. 其余 → UNRESOLVED_BOUND。

先单独输出交通界，再输出与 variability 组合后的动态界。
在相同 b 下验证 C→M→A 的 guaranteed/possible-within 集合嵌套；
随 b 增大两种集合均不能缩小。所有数值以 float64 计算，1e-6 仅为源 float32 容差。
“WITHIN”只针对本候选动态表达，绝不等于 hard FEASIBLE 或可派单。
不再把原 crawl/stop 超限作为第二道过滤条件。

## 输出与解释

- 每日/全体、每 profile 的 101 点动态界曲线及基线交叉计数；
- 每条路线的 U/Z/上界、原动态状态及零预算诊断状态（不是部署政策）；
- 按原 crawl/stop 是否超限的暴露分布、UNKNOWN 与 MIXED、连续拥堵时长分布；
- 每日/每 profile/每个发生变化的零预算类别最多 2 个稳定哈希案例，仅供定位；
- 完整脚本、配置、来源 SHA、资源记录及简明报告。

现有最长连续拥堵时长是 congested+severe 的合并时长，可帮助描述 C 的暴露。
不能拿它当 M 的 severe-only 连续时长。本批不引入连续时长门控。
仅凭这些聚合预测不能判断红灯、接客停留、短暂停车是否被误判；
相关语义验证保持未完成，不用少数案例伪装准确性验证。

## 运行与停止

命令：`D:/anaconda/envs/stage0-valhalla/python.exe -m stage3.scripts.traffic_state_offline_comparison`
目录：当前 stage0-v6-valhalla worktree 根目录。
输入：batch2 route_overlap、解释附加表、冻结 profile；按日窄列读入。
输出：`stage3/output/traffic_state_offline_comparison/` 和同名 docs 目录。
CPU 单进程，BLAS 单线程，无 GPU，无全路网时间矩阵；1800 秒硬超时。
进度逐日报告，抽样 RSS；失败不自动重跑分析。
工程验收：样本对账、基线复算一致、上下界/嵌套/预算单调、冻结输入 SHA 不变。
本批停止在离线报告；选择预算/未知政策、替换生产 profile、动态对照均不自动执行。
