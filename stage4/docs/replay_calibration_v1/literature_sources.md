# 历史 completed-trip 回放校准：定向来源注释

检索日期：2026-10-06（Asia/Singapore）。这是为本轮回放诊断准备的有限方法检索，范围是请求时刻、未观测接驾/空驶、车辆空闲与重定位、初始化、司机班次/供给、需求缩放和路网行程时长。共纳入 6 篇已核实出版信息的同行评议论文和 1 组 FleetPy 官方软件说明；其中 Yang 等的正文仅能读到出版商段落预览，数值设置未核验。未开展系统综述，不报告无法复核的总命中数或 PRISMA 流程计数。

来源以出版商、作者/大学公开版本及官方代码库为主。下面区分“原文采用的设置”和“迁移限制”；这些注释不构成新实验或模型参数变更决定。阅读由 AI 辅助完成，不代表用户已读全文。未向外部服务上传项目数据，也未运行仿真或程序化引文验证器。

## 1. Alonso-Mora 等（2017）：明确使用 pickup time 代理 request time

Alonso-Mora, J., Samaranayake, S., Wallar, A., Frazzoli, E., & Rus, D. (2017). On-demand high-capacity ride-sharing via dynamic trip-vehicle assignment. *Proceedings of the National Academy of Sciences, 114*(3), 462–467. [DOI: 10.1073/pnas.1611675114](https://doi.org/10.1073/pnas.1611675114)。同行评议论文；本轮阅读[作者公开 PDF](https://autonomousrobots.nl/assets/files/publications/17-alonsomora-ridesharing-pnas.pdf)的相关主文段落。

- 原文定位：印刷页 464 的等待/总延误定义；页 465 的在线请求池、重平衡及 `Results` 前三段；页 466 的实验阈值段。
- 采用设置：历史上车时刻直接作为请求时刻；等待是模拟上车减请求时刻，最大等待取 120、300、420 秒，最大总延误为最大等待的两倍。每 30 秒批量分配。午夜按历史需求分布抽样初始车位；每轮把空闲车与未服务请求配对，最小化接近它们的总行驶时间。路网采用每日平均边时长。
- 迁移限制：这些是替代车队的反事实运营假设，并未恢复真实下单时刻、接驾过程或 HV 班次；不能据此把历史 boarding timestamp 认定为实测 request timestamp。作者公开 PDF 首页含 2018 更正：平均行驶距离图及相关 SI 项被修正；本注释不使用原版距离结果。

## 2. Santi 等（2014）：仅载客行程的可共享性，真实寻车/等待未知

Santi, P., Resta, G., Szell, M., Sobolevsky, S., Strogatz, S. H., & Ratti, C. (2014). Quantifying the benefits of vehicle pooling with shareability networks. *Proceedings of the National Academy of Sciences, 111*(37), 13290–13294. [DOI: 10.1073/pnas.1403657111](https://doi.org/10.1073/pnas.1403657111)。同行评议论文；出版信息由 [MIT 原始存档记录](https://dspace.mit.edu/entities/publication/1c1cb630-061a-47f7-95cd-30f1e5bb97eb)确认，方法段落读于[大学镜像的原文 PDF](https://cdanfort.w3.uvm.edu/csc-reading-group/santi-pnas-taxis-2014.pdf)。

- 原文定位：PDF 正文第 2–3 页的行程元组、`Results` 的 Oracle/Online 定义；第 5 页 `Discussion` 末段和 `Materials and Methods / Trip Data`。
- 采用设置：输入是实际 pickup/drop-off 时刻与位置。Δ 限制相对原始单程的上车及下车延误；在线版本另用 δ = 1 分钟约束候选行程起始时间接近程度，示例讨论 Δ = 5 分钟。只处理 nonvacant trips；低密度情景随机去掉部分车辆及其行程。
- 迁移限制：δ 是配对反馈窗口，Δ 是共享延误容忍，不能混同于从真实下单开始计的接驾等待上限。论文未模拟完整车队接驾/重定位链，且明确将真实寻车/等待时间列为尚未知数据；这里的 shareability 不是 completed-trip 回放的服务率校准目标。

## 3. Vazifeh 等（2018）：期望上车时刻约束的最小车队问题

Vazifeh, M. M., Santi, P., Resta, G., Strogatz, S. H., & Ratti, C. (2018). Addressing the minimum fleet problem in on-demand urban mobility. *Nature, 557*, 534–538. [DOI: 10.1038/s41586-018-0095-1](https://doi.org/10.1038/s41586-018-0095-1)。同行评议论文；本轮读[出版商摘要](https://www.nature.com/articles/s41586-018-0095-1)及[作者公开的首个正文页](https://senseable.mit.edu/papers/pdf/20180524_Vazifeh_AddressingMinimum_Nature_1st-page.pdf)，未取得完整付费方法正文。

- 原文定位：印刷页 534 后半页，行程四元组定义及 minimum fleet problem 的正式定义；出版商摘要的驾驶员可用性限制段。
- 采用设置：给定期望 pickup time，要求车辆在该时刻或此前到达；真实数据的元组沿用实际 pickup/drop-off 时刻。路网行程时长由 taxi GPS 数据估计并考虑逐小时交通变化。全局最优版本需要离线知道当天需求，另提出近似在线版本；本轮未核验其在线视野等数值。
- 迁移限制：该零额外上车延误约束不等于原始乘客在现实中零等待。作者明确指出司机可用性约束可能提高所需车队规模。它支持检查行程衔接和定位供给缺口，但不提供恢复 HV sessions 的规则或固定 idle hold/预热时长。

## 4. Tachet 等（2017）：按时空需求分布缩放，逐小时估计行程时长

Tachet, R., Sagarra, O., Santi, P., Resta, G., Szell, M., Strogatz, S. H., & Ratti, C. (2017). Scaling Law of Urban Ride Sharing. *Scientific Reports, 7*, Article 42868. [DOI: 10.1038/srep42868](https://doi.org/10.1038/srep42868)。同行评议论文；本轮阅读[出版商主文](https://www.nature.com/articles/srep42868)的 `Introduction`、`Methods` 和相关缩放定义。该文发表于 Scientific Reports，并非 PNAS。

- 原文定位：`Introduction` 第 4 段的 Δ 定义；`Methods` 中预处理后的行程元组、随后 travel-time、subsampling 与 dynamic supersampling 段。
- 采用设置：载客行程的 pickup/drop-off 构成输入，逐小时估计路口间时长；代表性的共享延误容忍为 Δ = 5 分钟。降低需求时均匀随机抽取实际行程；提高维也纳需求时用 OD 概率、实测小时份额和 Poisson 过程生成额外行程，保留相应日内结构。
- 迁移限制：它研究行程的潜在可共享性，不是带司机会话、接驾和重定位的运营回放；这里缩放的是需求行程，不能直接充当同幅度供给缩放的依据。Δ 同样不是实测下单至接驾的耐心阈值。

## 5. Yang 等（2020）：匹配间隔、半径和耐心的权衡

Yang, H., Qin, X., Ke, J., & Ye, J. (2020). Optimizing matching time interval and matching radius in on-demand ride-sourcing markets. *Transportation Research Part B: Methodological, 131*, 84–105. [DOI: 10.1016/j.trb.2019.11.005](https://doi.org/10.1016/j.trb.2019.11.005)。同行评议论文；出版信息及摘要由[出版商](https://www.sciencedirect.com/science/article/pii/S0191261518311731)和 [HKU 作者机构存档](https://hub.hku.hk/handle/10722/308802)交叉确认。

- 阅读范围/定位：正式摘要、出版商 `Research problem` 与 `Conclusions` 段落预览；SSRN 公开稿入口未成功提供可读 PDF。
- 直接支持：平台累积等待乘客与空闲司机的匹配间隔，以及最大接驾距离/匹配半径，是两个独立控制变量。延长间隔可能降低接驾距离，但增加等待、诱发放弃；缩短半径可能降低接驾距离，但减少成功匹配。优化与供需水平相关。
- 迁移限制：本轮未读到完整方法，不能核验其实际耐心分布、固定数值等待上限、跨轮保留逻辑或历史时间戳转换。只用它支持机制权衡，不引用数值参数，也不以摘要补造参数。

## 6. FleetPy 正式论文（2026）：请求/供给/路网模块分别明确

Engelhardt, R., Dandl, F., Syed, A. A., Ding, C., Alvarez-Ossorio Martinez, S., Zhang, Y., Hamdy, H., Brodersen, J., Chen, Z., & Bogenberger, K. (2026). FleetPy: an open source simulator for reproducible research on mobility-on-demand services. *European Transport Research Review, 18*, Article 52. [DOI: 10.1186/s12544-026-00823-3](https://doi.org/10.1186/s12544-026-00823-3)。同行评议、开放获取；本轮读[出版商原文](https://link.springer.com/article/10.1186/s12544-026-00823-3) §§2.2.1、2.2.3、2.2.5.4、2.2.5.6、2.4、3.1–3.2、4。

- 采用设置：请求输入时间表示请求被提出的时间；time-based fleet sizing 激活/停用车辆，以表示驾驶员班次。重定位可按分区预测需求执行。Manhattan/Chicago 示例把最快路时长与历史载客行程时长比较，按小时求缩放因子。§4 示例最大等待 480 秒，boarding/alighting 各 30 秒/乘客。
- 迁移限制：论文对历史 TLC 时间戳是否回推真实 booking time 未作同等明确说明，不能据此断言它恢复了下单时刻。小时平均时长对齐不保证每条接驾路线正确。示例阈值和模块选择是研究设置，不是统一标准；所读段落未给固定 idle hold 或统一预热长度。

## 7. FleetPy 官方软件说明：字段语义和初始化接口

TUM-VT. (n.d.). *FleetPy Wiki: 3_input_data; Input_Parameters.md*. GitHub. 访问日期 2026-10-06。[官方输入说明](https://github.com/TUM-VT/FleetPy/wiki/3_input_data)、[官方参数表](https://github.com/TUM-VT/FleetPy/blob/main/Input_Parameters.md)。官方软件资料，非同行评议论文，无论文 DOI；本轮读取在线 wiki 与 main 分支参数表，未固定代码提交版本，也未运行代码。

- 精确定位：wiki `trips / requests` 定义 `rq_time` 为请求对系统变为 active/visible 的秒数，并另列 earliest/latest pickup 与 latest decision 字段；`active vehicles` 单独提供 time、share_active_fleet_size。
- 初始化/时间：wiki `initial state` 描述上段仿真终态及可能的下段初态；`initial vehicle distribution` 定义按节点概率抽样初始位置；`Network Dynamics Files` 单独定义按时间加载边时长或全局倍率。参数表分别提供最大等待、重定位方法/视野、动态 fleet-sizing 方法。
- 迁移限制：这些字段和接口提供可区分的建模语义，不能代替实际班次、真实订单创建时间或未知空驶轨迹的观测。官方在线说明会更新；若用于复现实验，应对应实际采用的软件版本。未从所读说明核验到通用 idle hold 或预热数值。

## 检索记录与覆盖限制

### 补充：网络时空校准方法线索（主agent复核）

Syed, A. A., Zhang, Y., & Bogenberger, K. (2023). Data-driven spatio-temporal scaling of travel times for AMoD simulations. In *2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC)* (pp. 3583–3588). [DOI: 10.1109/ITSC57777.2023.10422313](https://doi.org/10.1109/ITSC57777.2023.10422313)。出版信息由上述FleetPy正式论文参考文献29及[作者主页](https://yunfei-zhang.github.io/)交叉支持；研究目的与摘要由[作者arXiv记录](https://arxiv.org/abs/2311.06291)核验。

- 本轮实际阅读仅元数据与摘要；arXiv HTML/PDF获取失败，不标作已读方法全文。
- 摘要直接支持：以行程数据进行link-level路网时长的时空缩放，不要求额外实测网络状态；校准后的路网时长会影响AMoD仿真结果。
- 可作后续路线：从全城单一时段倍率，向有支持度的时空交通校准推进。但本轮没有核验其优化公式、正则项、数值参数，不能直接照抄实现，也不能把它当作获得真实接驾标签的办法。
- 补充检索词：`"Data-driven spatio-temporal scaling of travel times for AMoD simulations"`。这是一条额外方法线索，不改变以上6篇主检索论文的阅读覆盖声明。

纳入准则在检索前按任务限定：直接描述上述回放假设的一手论文或官方软件资料，优先可读方法原文；排除只转述结论的新闻、第三方总结及与该回放口径无关的工作。老论文作为明确的方法先例保留，不用于声称目前平台的实际策略。DOI/题名去重；未调用 Semantic Scholar/OpenAlex/Crossref 的 Python 客户端。

实际使用的检索词（公开网页搜索；引号为检索词的一部分）：

```text
Alonso-Mora 2017 On-demand high-capacity ride-sharing via dynamic trip-vehicle assignment request pickup time 7 minutes rebalance
FleetPy modular open source simulation tool mobility on demand services paper DOI travel time calibration warm up
Vazifeh Santi Resta Strogatz Ratti 2018 minimum fleet problem on demand urban mobility waiting time pickup time
Yang matching time interval matching radius on demand ride sourcing markets passenger patience DOI
Santi Resta Szell Sobolevsky Strogatz Ratti 2014 Quantifying benefits vehicle pooling shareability networks pickup times travel calibration
Tachet 2017 ride sharing PNAS Scientific Reports scaling law urban ride sharing demand pickup time
FleetPy github documentation demand data request_time active fleet driver sessions warm-up repositioning
"Quantifying the benefits of vehicle pooling" site:pmc.ncbi.nlm.nih.gov
"Addressing the minimum fleet problem" "pick-up" "delay" nature
"Optimizing matching time interval and matching radius" pdf Yang Qin Ke Ye
site:github.com/TUM-VT/FleetPy "Fleet Size" "TimeBased"
"Optimizing matching time interval" "patience" Yang pdf
"Optimizing matching time interval and matching radius" site:repository.hkust.edu.hk
"Optimizing matching time interval and matching radius" site:hub.hku.hk
"Vazifeh" "minimum fleet" "δ"
"Addressing the minimum fleet problem" filetype:pdf -site:minicod.com
"FleetPy" "warmup"
"FleetPy" "warm-up"
"Addressing the minimum fleet problem" site:arxiv.org
"Quantifying the benefits of vehicle pooling" site:senseable.mit.edu/papers/pdf
"Optimizing matching time interval and matching radius" site:sciencedirect.com "patience"
"FleetPy" "warm-up period"
"ride hailing" "trip chaining" site:nature.com/articles/s41467
"Quantifying the benefits of vehicle pooling" "δ"
"Quantifying the benefits of vehicle pooling" "online" "one minute"
"Quantifying the benefits of vehicle pooling" "doi" site:dspace.mit.edu
"Optimizing matching time interval and matching radius" "42" "pdf"
```

阅读层级限制：未全面阅读各文 SI；Vazifeh 限于首个正文页和摘要；Yang 限于摘要及段落预览；其余只核验注明的相关主文/软件段落。PMC 遇浏览器验证，部分作者 PDF 超时，因此改读出版商或公开大学镜像；不把失败获取记为已读全文。有限 Nature Communications 行程链检索未找到足够直接且已阅读的候选，未强行纳入。没有找到可据以确定本项目通用 warm-up 或 idle hold 时长的直接证据；这是本轮阅读覆盖的限制，不是证明这些方法不存在。

覆盖分布提示（DISTRIBUTIONAL_SKEW_ADVISORY）：6 篇论文中 5 篇发表于 2014–2020 年（5/6）；它们主要是仿真/算法方法来源，且多使用出租车载客记录。这是任务范围带来的覆盖集中，不能声称已覆盖网约车真实订单取消、司机班次恢复与实际空驶行为的全部文献。100% 服务率不是这些不同口径论文共同规定的校准标准。
