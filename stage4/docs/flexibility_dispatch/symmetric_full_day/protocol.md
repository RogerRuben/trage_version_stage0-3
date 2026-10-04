# 对称历史路线研究接口与M全天配对协议

## Material Passport

- academic-research-suite / experiment-agent；2026-10-04；run；版本 symmetric_full_day_v1。
- 本协议在新增native全天结果之前提交。输入派生不是新增策略实验。
- 用户授权：修正C/M/A与HV定义/接口不一致，随后执行此前计划的M全天两策略比较。

## 1. 研究定义

在共同数据支持与道路约束下，路线能力集合满足：

\[
\mathcal O_C\subseteq\mathcal O_M\subseteq\mathcal O_A=\mathcal O_{HV}.
\]

A是human-equivalent的研究能力上界，不是实际AV车型的安全认证。
乘客接受、车辆位置、可用时段、接驾到达截止时间属于dispatch层，因此A/HV实际服务量仍可不同。

历史反向overlay不再自动触发AV专属否决。按封存历史行驶方向建立独立virtual directed identity，反转边界INCOMING/OUTGOING及切向；不能将正向edge当作实际反向遍历，不能借用正向movement/restriction。
只为有冻结物理几何参考的历史身份提供这层研究表示；不重新导图，不重新clustering，不改699处封存信号数据，不允许geometry nearest-edge猜测。
无可用几何、未解析身份、必要预测输入缺失是共同输入排除，不是C/M/A能力不足；certified prohibited movement同样作用于HV/AV。
封存的AV routing topology、旧Stage3 profiles与原41场景均保留。本接口明确采用历史观测方向可用于研究回放的假设，不宣称OSM方向冲突均为地图错误或现实合法行驶。

C要求信控左转，不允许环岛/掉头；M允许非信控左转及环岛、不允许掉头；A/HV不额外拒绝这些maneuver。
C环境外状态为MR/CG/SC，M为SC，A/HV为空；预算保持5%。U作为描述、不自动否决，ML不重复叠加拥堵约束。
A/M/D/L及speed、CV/acceleration仍保留连续描述；其旧静态route聚合在反向区段可能不完整，不作为本次准入约束或完整路线复杂性真值。Gamma、cost、repositioning关闭，不叠加旧的同源crawl/stop约束。

## 2. 输入派生与最小核对

全Test31和20161010/17/24逐日处理，Arrow完整order批次流式解析反向movement。
Train需求库扩展至这三个日期的全天；仍沿用既有抽样multiplier=3、seed、三场景、future K5与2km，不用Test31未来请求拟合需求。
busy剩余时长模型直接复用上一轮3,400个Train配对的封存模型，不重新拟合；其训练覆盖是原两个时间带，不声称已验证所有时段的准确性。
M3预测、Train CDF、动态caps保持不变。这里只重建research compatibility标签，不重训任何stage。
必要核对只有反向LEFT/UTURN等语义、共同禁令、A/HV集合相等、C/M/A嵌套与native时间/数量对账。

## 3. 唯一两组native条件

- Test31全天，M，qA=0.5，passenger acceptance=0.7，原稳定CRN与车队抽样seed。
- MYOPIC vs SERVICE_PRESERVING_LOOKAHEAD；均从0秒干净native状态开始，不沿用旧不对称策略prehistory。
- 原30,000订单全部对账；共同输入排除单独列出。主服务率分母为共同模型population，同时给出原30,000单分母的服务率。
- 30秒rolling、300秒到达patience、current TopK20、确定性single-source matrix，实际native物理进展与drain不变。
- 固定172,800秒行政drain上界，全部已接受任务完成即可提前结束；不根据Test31未来实际时长设置优化器中的AV availability终点。日vehicle-hours仍只按24小时归一化，停止派单后不再引入新需求。
- 新策略固定critical→current service→carry-over最优面，再最大化预期下一服务数、最后最小接驾ETA。不是个体订单保护或全天优势保证。
- 每车每情景最多一次下一服务；未来ETA仍是Train预测OD-chord pace近似，不是完整随机VRP。
- 不新增C/A动态条件，不重跑18窗口或41场景，不作消融网格和结果驱动调参。

## 4. 资源与停止条件

两组顺序运行，各自独立CPU进程，OMP/OpenBLAS/MKL=1，CUDA禁用。
每轮20,000变量/150,000非零元/10秒上限，既有同状态MYOPIC资源回退记录保留；单组3小时硬超时，RSS>2,048MiB报告提醒。不构建订单×车辆稠密矩阵，不跨日期累计token表。
完成两组后只分析服务数、gained/lost、等待、AV/HV分工、资源/回退及物理对账；无论是否有优势均到此停止。
Test31已用于开发，结果称内部全天配对验证，不称未见Test或真实部署因果效应。

## 5. 执行入口

```powershell
python -m stage4.analysis.symmetric_research_prepare
python -m stage4.analysis.symmetric_flexibility_full_day --fleetpy-root D:/pycodes/didi_xian_raw/.external/FleetPy --policy MYOPIC
python -m stage4.analysis.symmetric_flexibility_full_day --fleetpy-root D:/pycodes/didi_xian_raw/.external/FleetPy --policy SERVICE_PRESERVING_LOOKAHEAD
```

输入派生发现Train非完整订单的预测时长不可聚合时已显式停止；修复为沿用冻结complete-dynamic订单集合，绝不逐token丢弃或以实际时长补值。`--resume`只复用已经原子落盘的Test31标签，再继续尚未完成的Train派生。

输出：`stage4/output/symmetric_flexibility_v1/`；说明与轻量结果：本目录。
旧结果不覆盖。完成后按用户已有授权提交、推送origin。

## 6. 用户授权的行政时限例外（2026-10-04）

用户明确允许：仅当第二组触发原3小时整组行政时限，保留该中止记录，以`--administrative-timeout-s 21600`从干净初始状态重跑同一第二条件。
MYOPIC不重跑；不新增科学条件；每轮10秒、20,000变量/150,000非零元、数据、seed、策略、质量及兼容阈值全部不变。配置文件及输入SHA不改。
这不是按服务结果调参。原进程继续使用已经加载的3小时时限，不尝试注入或热改进程；若它正常完成，则不执行此重跑。
