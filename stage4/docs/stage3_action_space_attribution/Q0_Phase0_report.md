# Q0 + Phase0 input closure report

2026-09-18；执行基线 `30a163cd9e22c08742054d915da151f9adc143e7`。

**Q0=PASS；Phase0=READY_AFTER_BOUNDED_PICKUP_ETA_COMPLETION。当前停止，等待补 ETA 与最终50条件的下一次授权。**

本轮没有任何 Valhalla 调用、service fallback search、M3 inference、动态模拟或 D0–D4 scientific MILP。Q0 有原语义的离线 MILP 复核；不能将“scientific MILPs=0”误读为本轮完全没用优化器。

## 1. Q0 reconciliation

使用已提交的十份完整 mixed arcs、旧 cumulative exposure/Gamma、原 P selected set，通过本轮 harness 调用同一已核验稀疏求解核心。十状态全部复现：前三项计数、P*、A1/A2的任意换配与 AV 指示改变的求解状态、最小 gap、五档预算存在性。

这是实现一致性 QA，不是额外科学条件，也不声称独立形式化最优性证明。新 Gamma-off D0 尚未求解，不沿用旧0/10。

## 2. Phase0 route closure

重新载入原登记的十个物理状态，state hashes全部一致；共720个不同等待订单。按订单过滤读取 original suitability/descriptor、final interface 和现有 fallback status/edges。

对其中373条已建立的 frozen bounded fallback，调用现有 Stage3 静态 evaluator 重建候选判定，随后仅使用 M profile。evaluator 内部计算其原有 profile 表，不运行 C/A 科学条件、不改 profile。未变更规则时的 fallback hard-feasible 结果与 frozen final 选路一致。

D1 从 original reasons移除reverse；D2只移除M的UTURN hard。原路线仍有其他hard才查既有fallback；所有其他unknown保留。特别检查原reverse可能遮蔽的 unresolved identity：用原descriptor的unresolved_token_count恢复未知标记，不因删reverse而误判为已解析。

本次涉及的720订单，D1/D2选路/静态状态均可按上述冻结数据判定：**COUNTERFACTUAL_ROUTE_UNRESOLVED=0**。这不等于所有反事实路线 FEASIBLE；正确判成 UNKNOWN/NONE 是一个已识别结果，而不是数据缺失导致无法判定。

fallback始终没有合法动态P50/rho补值。D3只放行结构已建立且hard FEASIBLE的最终路线；D4仅是资格松门对照。原路P50没有移植到fallback，任何candidate Valhalla service time都没有当作M3输出。

## 3. Condition-specific Top-K input counts

以下仅为需要取得权重的pickup输入规模，不是通过patience的弧数、matching capacity或assignment freedom。

| 条件 | required state-arcs | 旧同identity归档 | 未直接归档 |
| --- | ---: | ---: | ---: |
| D0 | 13099 | 130 | 12969 |
| D1 | 13243 | 130 | 13113 |
| D2 | 13110 | 130 | 12980 |
| D3 | 13272 | 130 | 13142 |
| D4 | 13696 | 117 | 13579 |

各条件共用同一空间状态/acceptance CRN，再各自shared Top-K20；同一条件内vehicle/request identity去重。跨条件union如下：

| 时点 | union required arcs |
| --- | ---: |
| 07:30 | 1350 |
| 08:30 | 1681 |
| 12:00 | 1829 |
| 13:00 | 1264 |
| 17:00 | 2016 |
| 17:30 | 2143 |
| 18:00 | 2317 |
| 18:30 | 1944 |
| 21:00 | 2014 |
| 23:00 | 1063 |
| 合计 | 17621 |

D4只覆盖旧130条中的117条也提醒我们shared Top-K并非跨条件单调；本轮不据这一数字推导实际HV排挤量或自由度，终层实验将按协议单列。

## 4. 真正需要补多少 pickup 查询

按state/vehicle/request身份：required=17621，directly archived=130，missing=17491。

进一步仅按现有adapter冻结cache规则复用——同一分钟/时区、source/pickup WGS84各7位、同beta/config/mode——可覆盖133条state-arcs；没有近邻插值。同一个routing key可以服务多个identity组合。

- 独立required routing keys：**16386**。
- 已有可认证权重keys：**125**。
- 独立missing routing keys：**16261**。

因此后续预计最多补16261个唯一pickup key，不能按五条件各自重复路由。绝不把缺失权重当作失败/无限ETA。旧routing样本未认证可扩展覆盖这些key，未拿不相符的样本降低缺口数。

旧routing config与beta SHA核对一致。复用权重绑定同一原physical state SHA；当前没有新交通数据或坐标变换。这里只统计缺口，尚未发起任何查询。

## 5. Fallback provenance：只到Top-K

十状态共有161个 `FALLBACK + hard FEASIBLE + evidence incomplete` state-order：

- 161个有至少一辆AV处于空间范围内（passenger前）。
- 114个在D3下有至少一条AV Top-K候选（包括原acceptance约束）。
- 最终通过pickup/patience数量：**NOT_RUN**。

这些是state-order计数，不等于全日6401条中的服务收益。进入Top-K不等于可及时接驾，更不等于存在A1替代解；不能据114推断evidence-driven结论。

## 6. QA、资源与归档

总Python执行约62.24秒，外部监护约64.15秒；250ms采样峰值RSS约785.64MiB，低于2GiB上限，无GPU。最大union状态2317条，未创建稠密order×vehicle矩阵。Phase0未调用MILP；Q0复用原求解器限时。没有触发资源上限。

三项小型route语义断言通过：移除reverse后保留unresolved identity；仅reverse时可恢复原路线；reverse与UTURN共现时不能一起被删除。核验50行输入计数守恒、union无重复、缓存同key权重一致、源config/beta hash一致。touched-file py_compile通过；不加新测试框架或全仓测试。

完整route counterfactual与pickup union保留本地可读JSON，并生成无损gzip版本，三份合计443313字节。安全检查未批准向当前origin上传包含订单/车辆标识、时间与坐标的明细，因此这些JSON/gzip及明细打包记录只保留本地，待用户明确批准后再处理。远端仅提交代码、文档与不含逐订单轨迹的Q0/Phase0汇总；日志不提交。没有删除原始产物。

## 7. 停止点

路由反事实可识别这一门槛已通过；pickup权重未齐，因此主实验仍未执行。**READY_AFTER...是输入状态，不是INVESTIGATE_STAGE3_FACTOR，也不是补路由的自动启动信号。**

当前不输出E/U/M/O新科学比较、A1/A2新结果或Stage3投入分支。下一步等待用户审查，明确授权后才补16261个唯一pickup查询并执行50条件；继续禁止service-route rerouting、M3训练/推理、全天FleetPy和冻结profile修改。

## 文件

- [冻结协议](stage3_action_space_attribution_protocol_v1.md)
- [Q0结果](../../output/paper_enhancement/stage3_attribution_phase0/q0.json)
- [Phase0汇总](../../output/paper_enhancement/stage3_attribution_phase0/phase0_summary.json)
- [缓存key闭合与QA](../../output/paper_enhancement/stage3_attribution_phase0/verification_and_cache_closure.json)
- [50行输入规模](../../output/paper_enhancement/stage3_attribution_phase0/condition_input_counts.json)
- 本地明细：`stage4/output/paper_enhancement/stage3_attribution_phase0/required_pickup_union.json`，未上传；同目录gzip亦仅本地。
- 本地归档SHA：同目录 `packed_artifacts.json`，未上传。
