# 端点影子身份对账：保留 clustering，不自动补 membership

## 结论

影子对账完成；目前**不能认证“仅补端点即可恢复 movement”的生产修复**。
没有修改生产接口、冻结网络、clustering、complex membership 或路线状态。
这一结果不是要求重新聚类，而是说明：仅凭现有精确身份与原生层级关联，
仍缺少新节点与冻结 physical-complex membership 之间的明确契约。

## 方法与范围

以 `cd52153` 的88条独立候选为输入，对两端共176个 endpoint-side记录对账：

1. 重新读取原生 edge owner / endnode。
2. 沿原生 NodeTransition 遍历；每一跳必须有反向 transition。
3. 仅以精确 GraphId 关联冻结节点表，再查原有 complex membership。
4. 同一 transition component 出现多个 frozen complexes 时标冲突，不任取一个。
5. 没有冻结节点锚点时保留未确定，不以附近坐标、OSM最近点或最小ID补齐。

这些 transition aliases 是影子层的物理位置关联，不会合并原生有向节点，
也不证明任意进入/离开该位置的 movement 合法或拓扑连通。

## 实测结果

| 对象 | 数量 | 解释 |
|---|---:|---|
| 缺失侧无 frozen node anchor | 88/88 | 涉及此前确认的55个原生节点；跨层级关联也找不到冻结节点 |
| 已有侧有唯一 frozen complex anchor | 87/88 | 69个身份一致端点，加18个明确跨层级关联端点 |
| 已有侧无 frozen node anchor | 1/88 | 前轮识别的同层错误身份；旧 complex 归属无法保留 |
| 多个 frozen complex anchor 冲突 | 0/176 | 不代表缺失节点已经可归属 |
| 原始 unresolved movement keys | 117 | 分母保持 |
| 原始 unresolved encounters | 391 | 全部逐项留存，未重解析删行 |
| 涉及无锚点原生端点的 encounters | 391 | 全部保留 UNKNOWN |
| 涉及同层错误身份/旧 complex 不保留 | 2 | 单列，不能作为18个跨层级别名处理 |
| 原记录链不连续 | 2 | 独立标记；与上一行不预设同一集合 |

实际恢复 movement = 0。这是保守对账结果，**不是证明这391条路线现实不可通行**。

## 为什么暂不修生产接口

不能使用以下捷径：

- “节点没在membership里，所以一定是complex外部节点”；
- 把新增 native node 默认为旧 complex 成员；
- 删去缺失端点的boundary记录后，把消失的encounter算作通过；
- 把已确定的18个层级alias当作88条完整movement均已恢复。

这些做法都会加入尚未验证的 membership 语义。唯一锚点的87个已有侧可以
成为以后最小接口修正的依据，但它们不足以单独闭合任何本轮问题movement
的全部证据。故本轮不制造一个名义上的“生产修复”。

## 下一步的最小决策

若继续争取不重新聚类，下一步应限定为**冻结complex下的新增节点拓扑归属审查**：
明确哪些证据足以判定新原生节点为旧complex的内部、外部或边界，保留无法判定项。
这可以先在影子层完成，不必全网重聚类。若证据要求改变已有成员或合并/拆分complex，
再单独提出授权；不能把这种变更藏在端点补齐里。

本轮不自动启动这项语义扩展，不修改reverse、fallback branching、C/M/A、M3、
41场景或论文结论，也不运行固定状态assignment。

## 交付与资源

- [176行影子对账表](endpoint_shadow_reconciliation/endpoint_shadow.csv)
- [汇总与源文件哈希](endpoint_shadow_reconciliation/summary.json)
- [QA](endpoint_shadow_reconciliation/qa.json)
- 工具：`stage4/tools/reconcile_native_endpoint_shadow.py`
- 私有逐encounter账本：`stage4/output/paper_enhancement/endpoint_shadow_reconciliation/encounter_reconciliation_private.json`

公开表仅含静态网络身份，不含订单/车辆/轨迹信息。逐订单账本保留在本地忽略目录。
单CPU进程约0.69秒，峰值工作集227.30 MiB，无GPU、routing、matrix、推断、优化器调用。
冻结边、节点、membership、movement、boundary及读取的原生tiles前后哈希一致。
