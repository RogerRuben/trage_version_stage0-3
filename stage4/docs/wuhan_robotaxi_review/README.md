# 武汉 Robotaxi 公开数据获取与项目启示

日期：2026-09-25。项目检查基线：`a7babf0`。本轮只下载、检查和分析公开材料，没有修改冻结模型、Stage3 profile、候选门、41 场景或论文正文，没有运行第三方仿真代码。

## 1. 结论

这篇论文对我们最有价值的地方，是补充真实运营环境的外部证据，并帮助拆开三个容易混淆的对象：**在哪里提供服务、车辆何时能够服务、具体道路动作能否通过**。

公开数据比聊天后半段描述的更丰富：除了用于随机 AV/HV 渗透仿真的 7 月订单，还有 11 月 Robotaxi 处理样本及论文图表 Source Data。后者提供真实请求生命周期字段和运营活动统计。不过，这仍不是可用于重建真实 AV movement 的逐点 GPS 数据集。

建议优先使用 11 月样本核对 timing 定义，使用图表源数据支持空间异质性讨论。SNE 可作为只读的备选描述量；不应立即用它替换 A/M/D/L，更不能用它放开方向或 UNKNOWN 门。

## 2. 来源与本轮可访问范围

论文：Wang et al., *Urban energy and transport impacts of autonomous robotaxi deployment*, Nature Sustainability，2026-09-22，DOI `10.1038/s41893-026-01944-2`。

- [期刊页面、摘要、Data availability](https://www.nature.com/articles/s41893-026-01944-2)
- [公开仓库固定版本](https://github.com/Tlab-seu-code/Urban_Robotaxi_Analysis-/tree/d8de2c56e6c127542c98c9d224021cd131cd4f77)
- [补充材料](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41893-026-01944-2/MediaObjects/41893_2026_1944_MOESM1_ESM.pdf)
- [图4源数据](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41893-026-01944-2/MediaObjects/41893_2026_1944_MOESM12_ESM.xlsx)

期刊正文当前有订阅限制。本轮核对的是公开摘要、补充材料、源数据与作者代码，不声称完成全文方法复核。完整运营记录需要向通讯作者申请并签 DUA；没有擅自联系作者或申请账户权限。

用户提供的聊天是待核对的分析材料，不是本轮执行实验的授权。没有依据聊天建议运行 SNE、gate ablation 或新 Stage4 场景。

## 3. 已经下载到本地的内容

统一目录（在工作树中，Git 忽略）：

`stage4/output/external_research/wuhan_robotaxi_2026/`

下载清单记录 80 个文件，合计 849,955,508 字节，约 850 MB（811 MiB）；目录另有少量元数据。没有把 LFS 指针误当数据，三个大文件都核对了原 LFS SHA256。

| 产品 | 大小（十进制） | 检查到的实际内容 | 合理用途 |
|---|---:|---|---|
| Wuhan-sample 的 `traj_20240708.csv` | 127.43 MB | 180,102 条唯一订单，11,248 辆车；OD、起止时间、SUMO 边与 path | 仿真输入、接口测试，不是真实 AV/HV 对照 |
| 武汉 `robust.net.xml` | 288.17 MB | SUMO XML 路网；LFS 哈希通过 | 作者方法复现底图，不能直接当 Valhalla canonical graph |
| `v12-traj_sample_2024-11-12.csv` | 3.01 MB | 6,727 条唯一完成订单，402 辆车；37 字段 | 请求时间/接驾时间语义诊断；处理后的 Robotaxi 订单样本 |
| `speed_filled_step1_sample_2024-11-12.csv` | 333.86 MB | 10,781,328 行；hour/id_y/weekday/avg_speed | 路网小时速度输入；填补速度不等于直接观测标签 |
| MOESM1 PDF | 1.72 MB | 12 页补充材料 | 研究期、补充分析定义 |
| MOESM2–12 XLSX | 95.03 MB 左右 | 11 个图表源数据工作簿 | 已发表统计和聚合时空活动分析 |
| 作者代码、README、小型结果摘要 | 约 1 MB | 清洗、SNE、路由、匹配、图表代码 | 只读方法核对，未安装其依赖或执行代码 |

完整文件路径、URL、大小和哈希见本目录 `download_manifest.json`；表结构与只读统计见 `inspection_summary.json`。这些是元数据副本，不含订单/车辆/用户标识。

### 本轮明确没有下载或获取

- 全国 POI ZIP（863,121,587 字节）与夜光 GZ（328,397,352 字节）：已核对 release 元数据及链接，但为控制磁盘占用，未下载与当前核心问题关系较弱的约 1.19 GB 附件。
- NYC/Porto 大型数据及路网、重复武汉路网、pickle 结果：未纳入有效下载清单。发现阶段有少量其他城市说明/指针，不可作为完整数据交付。
- 非公开全量 Robotaxi/HV 记录、真实逐点轨迹、逐道路能力标签、合法运营边界及版本历史：未取得。
- 原作者目录树未见 LICENSE 文件；公开访问不等于可以不受限制地改发所有数据。本轮仅在本地保留原文件，远端只提交自己的说明、脚本和无标识元数据。样本含用户标识和文字地址，不在报告中展示或重新发布。

## 4. 聊天中哪些判断成立，哪些需要再修正

### 4.1 成立：7 月 AV/HV 标签是仿真分组

[`split_by_vehicle_sample.py`](https://github.com/Tlab-seu-code/Urban_Robotaxi_Analysis-/blob/d8de2c56e6c127542c98c9d224021cd131cd4f77/7-sample_data_and_code/Wuhan-sample/code/split_by_vehicle_sample.py) 对 vehicle list 随机打乱，按比例生成 AV/HV 文件。样本没有真实车型列。

还有一个复用细节：同一个 ratio 的 `*_av` 和 `*_hv` 都选该比例的车辆，不能直接拼起来当 p/(1-p) 的车队。调用者须按仿真配置正确配对互补比例。不要只靠文件后缀判断总体构成。

### 4.2 成立：不能把 route 字段自动认作实测轨迹

[`generate_routes_sample.py`](https://github.com/Tlab-seu-code/Urban_Robotaxi_Analysis-/blob/d8de2c56e6c127542c98c9d224021cd131cd4f77/7-sample_data_and_code/Wuhan-sample/code/generate_routes_sample.py) 从 OD 匹配最近边，再求 passenger shortest path。实际下载 CSV 已含 path，但没有逐点 GPS、map-matching 质量或路径来源证书；不能仅凭 path 验证真实 U-turn/roundabout/unsignalized-left。

11 月 v12 表只有起终点与端点边，没有逐点 GPS 或完整实测路段序列。它的请求字段可用，不意味着 movement 证据可用。

### 4.3 修正：真实运营观察并非只能确认 7 月

补充材料明确标明真实 Robotaxi 观察期为 2024-11-12 至 12-12。7 月 8 日属于另一个公开渗透仿真输入，二者必须分开登记。

### 4.4 修正：不能说“真实 Robotaxi 订单字段只在代码里，公开样本没有”

11 月样本确实保留：呼单、接单、出发接驾、车辆到达起点、开始行程、到达目的地、订单状态等字段。它按到达起点日抽取，全部为完成订单。这比只有 OD 的 7 月表更有价值，但不是取消/未服务订单总体。

### 4.5 修正：图4不只有起点网格分析

[`fig4a.py`](https://github.com/Tlab-seu-code/Urban_Robotaxi_Analysis-/blob/d8de2c56e6c127542c98c9d224021cd131cd4f77/6-figure/fig4a.py) 读取 trip-level `mode, SNE_mean, dist_km`，绘制距离加权的复杂度直方图。Source Data 给出该聚合分布，而非逐点路径。这比“仅订单起点服务域”多了一层 operational-path proxy，但仍不能证明具体 maneuver capability。

图4a工作表同时给出 HV/Robotaxi 的距离加权均值 2.65/2.20；其样本计数与补充材料的完成订单总体不同。因此，不应把不同图的记录当作同一个筛选样本直接合并。完整定义和筛选差异需要正文或作者说明。源表 Welch_p 存为0只是发布数值的记录，不解释成数学上的精确零概率。

### 4.6 修正：SNE 不是一个天然、无歧义的 0–1 能力分数

[`compute_grid_sne_wuhan.py`](https://github.com/Tlab-seu-code/Urban_Robotaxi_Analysis-/blob/d8de2c56e6c127542c98c9d224021cd131cd4f77/5-cci_and_sne/compute_grid_sne_wuhan.py) 同时输出原始熵 `SNE=H` 与 `SNE_norm=H/ln(36)`。图4b加载代码优先读取原始 SNE，图4a发布统计也不是 0–1 尺度。不能把聊天公式、图值、profile cap 不加转换地拼接。

## 5. 本轮直接从样本计算的 timing 检查

这些是对公开处理样本的描述性统计，不是重新估计我们的模型。

| 完成订单时间差 | P25 | P50 | P75 | P90 |
|---|---:|---:|---:|---:|
| 呼单至车辆到达起点 | 180 s | 300 s | 480 s | 720 s |
| 呼单至开始行程 | 240 s | 360 s | 540 s | 744 s |
| 呼单至接单 | 0 s | 0 s | 0 s | 60 s |

全部 6,727 行可解析，上述时间差没有负值。呼单至接单有 5,908 行为零，主要需要考虑分钟级时间戳的量化，不能解释成系统真的零响应时延。开始行程的最大差值达 8,700 秒，应逐字段核实后再作分布外推，而不是本轮自行清洗。

关键陷阱：[`run_carpooling_one_day.py`](https://github.com/Tlab-seu-code/Urban_Robotaxi_Analysis-/blob/d8de2c56e6c127542c98c9d224021cd131cd4f77/7-sample_data_and_code/carpooling_one_day_sample_code/run_carpooling_one_day.py) 把“到达起点时间”赋给内部 `call_time`。这是算法时钟，不是真实呼单字段。若我们复用，必须另设 request_time、vehicle_arrival_time、boarding_proxy_time，不能沿用变量名猜语义。

这些分位数的最合适用途是说明 **request release 与 trajectory departure 之间确有实证差别**。它们不能直接估计潜在 patience：完成订单的 observed wait 是经历的等待，未观测的最大愿意等待时间至少是另一个随机量；缺失取消、未响应和选择过程时无法从完成样本识别其分布。

同理，不能把武汉 2024 的 P50=300 秒拿来宣布西安 2016 的 300 秒 patience 获得验证。恰好数值相同不等于对象相同。

## 6. 除聊天之外，对我们更重要的帮助与限制

### 6.1 应优先研究“服务活动选择”，而不是从活动缺失推断能力缺失

观测到的 Robotaxi 活动是技术能力、监管服务域、运营商部署、供给、价格和乘客需求共同产生的。某区域没有 AV 订单，不能直接标成 incapable；有 AV 经过则可反驳“所有真实 AV 都绝对不能经过”的泛化，但不能据此要求假设性 Conservative profile 全部通过。

若以后获取 geofence 和运营覆盖，优先在共同可运营区域内比较同时间段、相近 OD/距离/道路等级，报告条件分布、标准化差异与支持范围。不要直接把全市 AV-vs-HV 差异变成能力分类器。当前公开样本不足以完成这个控制。

### 6.2 Source Data 提供了比聊天建议更便宜的外部证据入口

图2b工作簿含 grid × hour 的 Robotaxi/HV count。表格以多个并排块组织，一张工作表有 1,000,001 行、17列；这不是“一百万原始订单”。图1还含网格活动及模型解释数据，图3有小时运营状态比例。

后续可以先拆解并核对聚合表，研究空间覆盖/活动重叠、零计数区分及时间变化。不必先做武汉全路网 map matching。但 source table 可能带结构性零、缺失块或不同口径，须核对可运营区域与支持集，不能直接以零 count 训练 hard eligibility。

只读查看采用流式工作簿读取，没有把百万行工作表整体加载为 DataFrame。本轮未进一步拟合空间模型。

### 6.3 SNE 的好处是补充形态，而不是替代瓶颈语义

局部最大值和整条路线平均值回答不同问题。一个必要动作即使只出现一次，也可能确实构成约束；平均值低并不证明最大值过于保守。因此建议保留 max、mean、tail exposure 的并列诊断，不立即把 max 换掉。

SNE也有自己的偏差：

- 方向分布规则不意味着 signal/control、车道数或冲突点简单。规则棋盘格依然可能有大量复杂路口。
- 作者按 edge 计数、用边中点归格、用首末点方位估计方向。canonical 切分次数、双向表示、边跨格、曲线简化均可改变结果。
- 代码先做 72 个5度 bins，旋转后成对合并为36个10度 bins；直接照聊天写36 bins不能保证数值一致。
- 网格尺度、相位、空格与低支持格必须明确；没有道路证据不能填 SNE=0。
- time-weighted exposure 会混入拥堵和预测器，distance-weighted更接近纯形态，intersection-count-weighted则是不同估计对象。跨城比较须先统一权重，不能把这些差异都叫 AV能力。

因此建议先建立“作者忠实实现”和“切分稳健版本”两个定义，后者必须明确命名为我们的方法。禁止为使 C/M/A 更好看而选择尺度或阈值。

### 6.4 当前仓库不支持“SNE替换会直接打开主场景动作空间”

本地 `stage3_action_space_attribution/A0_repository_audit.md` 记录：27个 MAIN 配置 Gamma=null，cost=false；有限 rho 超1不直接删弧，但非有限 rho 会造成 exposure 缺失并影响资格。

`d13_report.md` 则表明，剩余部分替换见证涉及 movement UNKNOWN、原路线/fallback分支及方向/证据语义。把静态 max 改成 SNE不会自动修好这些对象。用 SNE 填补缺失 rho、跳过 UNKNOWN，相当于修改 evidence policy，而不是简单更换 descriptor。

建议分开两条线：

1. 描述有效性：SNE 与 A/M/D/L 是否互补、max与整体暴露如何不同。
2. 决策作用：该描述量究竟经哪个已启用门、目标或约束影响弧/assignment；若没有启用路径，相关性变化本身不会产生控制收益。

这比先新增 `G_SNE`、再看匹配数是否上涨更可解释，也避免靠放松门制造改善。

### 6.5 可借鉴动态链式服务视角，但不直接搬 minimum-fleet 结果

作者 fleet 优化的图以 trip 为节点，以前单结束后能接后单为弧；我们的当前图以订单和车辆为两侧。理想 DAG 上 minimum path cover 的 `N−maximum matching` 与 instantaneous matching capacity 不是同一指标。

它可启发我们检查 assignment 后车辆释放地点/时刻与后续兼容需求，但不证明 lookahead 一定优于信息对齐的简单规则。当前两期研究已有 STOP_OR_SIMPLIFY 结论，不能因新论文用 matching 就自动重新启动复杂控制器。

如以后要建离线上界，必须固定需求集、接驾时间规则、服务域、信息集与可行弧完整性。作者代码含两小时连接筛选、空间阈值与后备规则；不能未经核对就称其实现为我们问题的严格数学上界。更不能将其车队缩减与我们的全天服务率直接作百分比差。

### 6.6 充电与有效供给值得讨论，但暂不加能源模块

公开小时状态比例说明 nominal在线、空闲可派、接驾、载客与充电应分开。它可以帮助限定我们的 full-horizon AV 假设，而不是用一个武汉平均充电比例直接缩放西安 AV-hours。

能源收益还混合自动化、动力类型、车型参数及共享/调度设定，不能直接归因为 AV 更聪明。当前论文主线是兼容约束与服务能力转换，加入 SUMO 能耗不会补足其关键识别问题。

### 6.7 预测验证不能照搬时序口径

补充材料采用 leave-one-day-out；这种方案允许用被留出日之后的日期训练。它适合相应稳定性问题，但不等价于我们要求的严格向未来 rolling-origin。保留我们的 train-only拟合、calibration及decision-time信息边界，不用其高 R²为 M3 decision superiority背书。

## 7. 推荐顺序（建议，不是自动执行授权）

| 优先级 | 最小下一步 | 产物 | 不做什么 |
|---|---|---|---|
| P0 | 对11月样本做字段字典和完成样本选择诊断 | timing定义、分钟精度、异常与可识别性说明 | 不拟合潜在patience、不改西安阈值 |
| P0 | 整理图2/4源数据与研究总体 | 可引用的外部证据表及口径差异 | 不把聚合活动当逐动作能力 |
| P1 | 西安静态descriptor只读敏感性方案 | 同一网络、同一route、max/mean/tail并列 | 不加SNE gate、不重跑41场景 |
| P1 | 若确需真实maneuver验证，起草小样本DUA请求 | 真车型、GPS、时间精度、geofence、版本、抽样说明 | 不要求完整用户标识，不擅自发送 |
| 暂缓 | 新fleet控制器、全量武汉匹配、SUMO能耗 | 等研究问题和授权确定 | 不把数据到手当作扩项授权 |

如果只能做一件后续工作，我建议先做 **11月真实请求生命周期字段的可用性闭环**。它直接对应当前已经确认的 RT-sensitive 限定，且公开文件足以开始，不必先增建一套武汉 Stage0–3。

## 8. 可复现性与限制

获取脚本：`stage4/tools/acquire_wuhan_robotaxi_public.py`。检查脚本：`stage4/tools/inspect_wuhan_robotaxi_public.py`。固定上游 commit，流式下载，单文件解析，无 GPU、无 order×vehicle 稠密矩阵。没有执行下载的 Python、PowerShell 或 pickle。

检查范围是文件到手、LFS完整性、CSV行数/唯一标识数量/字段、完成样本时间差、工作簿表结构与图4发布统计。没有证明公开样本代表完整运营总体，没有完成map matching、独立复现论文图表、对运营商身份推断或AV安全认证。

本轮为单篇论文聚焦分析：没有构造跨论文冲突清单，也不声称完成系统性文献综述。报告中的研究建议是我们结合本地代码与证据边界作出的判断，不是原论文结论。
