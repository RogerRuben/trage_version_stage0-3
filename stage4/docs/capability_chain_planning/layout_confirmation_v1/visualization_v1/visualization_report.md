# C初始部署研究：三张核心图与图用数据

## 1. 交付范围

- 日期：2026-10-14，Asia/Singapore。
- 状态：**D1_D2_D3_COMPLETE**；只读后处理与图形核查完成，不是新实验或独立重复验证。
- 按[设计规格](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/layout_confirmation_v1/visualization_design_v1.md)生成三张中文核心图，各含300dpi PNG与PDF。道路底图在PDF中按300dpi栅格化以控制体积，前景标记、路线差异、文字保留为向量。
- 采用academic-research-suite图形工作流；按PDF技能用Poppler渲染最终PDF进行可读性检查。核查由同一助手内联完成，不声称独立审稿或外部VLM验证。
- 未运行仿真、布局优化、M3推理、路由、外部地图请求或GPU计算；未改变策略、能力、供给、阈值、种子及原冻结结果。16份记录的输入文件在处理后校验未改变。

## 2. D1：部署变化与服务足迹

[查看PNG](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/layout_confirmation_v1/visualization_v1/d1_network_deployment.png) · [查看PDF](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/layout_confirmation_v1/visualization_v1/d1_network_deployment.pdf)

![D1 原10%参考的路网部署与服务足迹差异](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/layout_confirmation_v1/visualization_v1/d1_network_deployment.png)

**图注。** 原10%参考条件下，联合机会布局相对C兼容热点的空间变化。A表示上线位置的日历日C可接单小时差：面积对应绝对差，蓝色实心/橙色空心符号分别表示增加/减少。B表示C完成订单在已有有向路段上的覆盖数差：同一订单同一有向身份只计一次，线宽对应绝对差，历史反向身份以点线区别。C表示共同人口中新增和失去服务的请求起点，形状分别为圆点和叉号；共新增110单、失去65单，全队净增加45单。全部面板使用WGS84转EPSG:32649后的共同范围，含2km比例尺。两方案C可接单时间均为1,227.9322小时；订单位置、选定路线引用和图用覆盖几何无缺失。

读图说明：

- 按上线位置的整日净分布计算，有352.53小时从一些位置重新分配到其他位置，约为C车时的28.7%；不是新增供给、已执行调位时间或调位成本。
- 变化主要集中在图中南部的道路走廊：西南一带有较明显的车时和C覆盖减少，北侧、东侧的若干上线位置及道路覆盖增加。这里的“北侧”指变化集聚区域内部的相对位置，不是全市北部。
- 新增与失去服务位置存在重叠，因此不能把改进简单描述成扩大一个地理服务圈，或由某个局部热点独立造成收益。

覆盖与边界：

| 项目 | 实际数量 |
|---|---:|
| 共同请求人口 | 28,367 |
| 两方案C服务订单并集 | 1,517 |
| 去重的C订单-有向身份对 | 45,783 |
| C覆盖身份键 | 4,903 |
| FULL_NETWORK_EDGE覆盖键 | 4,439 |
| 历史反向overlay覆盖键 | 464 |
| 发生覆盖数变化的身份键 | 3,884 |
| 绘示的底图有向边记录 | 55,070 |
| 缺失请求位置／图用覆盖几何 | 0／0 |
| 共同人口缺失ORIGINAL引用／路线身份 | 0／0 |

4,903和3,884是有向身份键数量，不是独立物理道路条数。正反向及overlay统计分开，但在相同物理道路上可能视觉重叠。底图55,070也不是去重后的物理道路数。完整研究范围由共同请求路线与两方案上线位置联合确定，未只裁取收益大的城区。

这张图展示**被服务历史订单的路线足迹**，不是native逐边位置日志、实际交通流、GPS轨迹或路段载客利用率。edge几何是路段符号，含首尾partial的覆盖；反向几何来自已有明确绑定，不将其解释成AV方向认证。上述0缺失只针对本图的共同人口，不等于上游所有30,000单均无缺失。

## 3. D2：服务重新分工与资源转换

[查看PNG](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/layout_confirmation_v1/visualization_v1/d2_service_conversion.png) · [查看PDF](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/layout_confirmation_v1/visualization_v1/d2_service_conversion.pdf)

![D2 订单服务归属转换与载客增量账面分解](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/layout_confirmation_v1/visualization_v1/d2_service_conversion.png)

**图注。** A配对显示20%条件下28,367个共同请求的最终服务者，行对应兼容热点、列对应联合布局。灰色为归属不变，蓝色为C/HV换手，绿色为新增服务，橙色为失去服务；九格保留原始单数。HV转C为770单，C转HV为611单，未服务转C为6单，C净增加165单；全队新增113、失去128、净减少15单。B显示四个固定对照点的载客增量账面分解：蓝色为获服务窗口数量项，橙色斜线为条件载客强度项，黑色菱形为净增量；保留5%的负数量项。

数学口径沿用L=nm的对称恒等式。条件强度包含参与者组成、行程时长和订单数量，不等于续单次数，更不等于M3的独立因果贡献。订单换手不同于新增服务；这些结果也不是固定状态maximum matching或混合系统容量的可加分解。

## 4. D3：完整运营取舍

[查看PNG](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/layout_confirmation_v1/visualization_v1/d3_operational_tradeoffs.png) · [查看PDF](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/layout_confirmation_v1/visualization_v1/d3_operational_tradeoffs.pdf)

![D3 C载客利用与全队服务、空驶的配对增量](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/layout_confirmation_v1/visualization_v1/d3_operational_tradeoffs.png)

**图注。** 四个固定对照点的联合布局减兼容热点增量：A为C窗口内载客利用率，单位百分点；B为全队最终完成订单，单位单；C为空驶距离，单位km，负值表示减少。菱形标明复用的原10%参考，圆点为新增条件。C利用率在四点均改善，但5%空驶增加、20%全队完成量减少。没有独立重复，不画虚构误差棒、显著性星号、连续规模响应或帕累托前沿。

所有条件使用同一日期，q是基准归一化C车时水平而非简单车辆数比例。各对内部供给与在线控制相同，不能把跨规模四点当四个独立随机重复，也不能据此宣布最优商业规模。

## 5. 图用数据、复现与资源

可直接核对的匿名聚合：[四条件效应](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/layout_confirmation_v1/visualization_v1/effects.csv)、[载客分解](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/layout_confirmation_v1/visualization_v1/loaded_decomposition.csv)、[全部36格转换计数](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/layout_confirmation_v1/visualization_v1/service_transitions.csv)。

空间小表保存在忽略目录stage4/output/capability_chain_planning/layout_confirmation_v1/visualization_v1，包含deployment_delta.csv、route_coverage_delta.csv、service_change_origins.csv和map_extent.json，不上传逐订单位置或窗口记录。数据来源记录与覆盖量见[data_summary.json](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/layout_confirmation_v1/visualization_v1/data_summary.json)，最终图件、代码SHA和核查记录见[render_summary.json](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/docs/capability_chain_planning/layout_confirmation_v1/visualization_v1/render_summary.json)。

两个独立进程依次执行，未跨条件累积大表：

| 工作 | 实际程序秒 | 峰值MiB |
|---|---:|---:|
| 图用数据后处理 | 55.13 | 248.42 |
| 首轮绘图 | 48.48 | 291.04 |
| 一次版式修正后的最终绘图 | 39.29 | 290.49 |

三项合计约142.90秒，不包括工具调用、PDF复核和人工读图的日历时间。这是后处理耗时，**不能解释成全天仿真被加速到几分钟**。所有峰值低于规格的约512MiB目标，无GPU、外部底图下载或订单×车辆×时间大矩阵。

从相同既有源数据重新生成到空目标目录：

1. 使用D:/anaconda/envs/stage0-valhalla/python.exe，在工作树根运行模块stage4.analysis.scheme_a_visualization_data，参数为--root .。已有data_summary时拒绝静默覆盖。
2. 运行stage4.analysis.scheme_a_visualization_figures，参数为--root .。--replace仅用于本批自有图件的版式修正，不能覆盖冻结实验。

[数据代码](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/analysis/scheme_a_visualization_data.py) · [绘图代码](D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla/stage4/analysis/scheme_a_visualization_figures.py)

## 6. 核查与图形—结论对应

数值核查通过：四张矩阵总和均为28,367；行列边际与各组C/HV及全队完成量一致；有向覆盖键无重复，差异等于J-H；图用CSV没有逐订单ID、车辆ID或坐标。源码编译和diff check通过。

三份最终PDF均为单页，Poppler渲染成功，并用bundled pypdf确认中文字体已嵌入。首轮发现D1底部说明重叠、D3边缘数值标注拥挤；只调整留白及标注位置，经一次修正后检查通过，无剩余已发现的截断或字体缺失。核查用临时PNG已移除，可由最终PDF重新生成。不增加独立审计体系、性能Gate或实验。

figure_table_trace（呈现对应，不是新的evidence bundle）：

| artifact_id | source_data | transformation | caption_claim | supported_manuscript_claims | limitations |
|---|---|---|---|---|---|
| D1 | 空间三张小表、既有路线身份和道路几何；路径见第5节和data_summary | 数据/绘图模块；SHA见data_summary与render_summary | 原10%部署和C服务足迹发生空间重新分布，全队新增110、失去65 | 空：本步不改稿，尚未向正文注册Figure编号或引用 | 整日聚合；非GPS/车流；方向图形可能重叠；不是局部因果 |
| D2 | service_transitions.csv、loaded_decomposition.csv | 同上 | C/HV换手和新增服务不同；载客增量的数量/强度来源因条件不同 | 空：未修改正文 | 账面恒等式，不是模块消融；窗口不是商业车辆 |
| D3 | effects.csv | 同上 | 四点C利用改善，但全队服务与空驶存在正负取舍 | 空：未修改正文 | 单日、少种子、无独立重复，不识别最优规模 |

交付到此停止。不追加仿真、能力/规模网格、预测消融、路线重规划、动画系统或论文自动改写。
