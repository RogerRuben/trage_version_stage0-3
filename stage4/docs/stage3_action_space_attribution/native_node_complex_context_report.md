# 55个原生节点相对冻结complex的拓扑归属审查

## 结论与边界

完成55/55个原生节点的只读审查，未重新聚类，未修改membership或生产接口。
这些GraphId对应51个原生双向层级transition关联组。它们的身份和局部拓扑
已知，但冻结产品没有为它们提供physical-complex归属；不能把“无记录”解释为
“一定在complex外部”，也不能以最近complex直接补membership。

本轮将问题进一步分为近邻归属审查和远端道路/路口语义，不再将55个节点视为
同一种简单缺失端点。没有认证任何新的movement或route FEASIBLE。

## 冻结算法口径

读取 `stage3/odd_tod/intersection_complex.py::construct_membership` 确认：

- 10m tolerance下，candidate之间的空间pair门槛是20m，不是10m。
- 合并还涉及grade与roundabout约束，随后按诱导拓扑拆分。
- 多candidate组使用10m候选点buffer搜集连接节点；单candidate组并不会直接
  将buffer内所有节点纳入。

本轮没有调用candidate detection、construct_membership、query_pairs或UnionFind。
只读取既有candidate members，按其精确GraphId读取原生坐标，建立一个KD-tree。
距离采用冻结算法同一投影EPSG:32649。距离用于关系诊断，不用于身份匹配或归属。
这些距离是到冻结candidate节点的距离，不是到complex多边形边界的距离，也不是
按原MVT量化坐标重新执行聚类。因此阈值附近仅能标记审查，不能宣布应合并。

## 主要分类

| 与冻结candidate及complex的关系 | GraphId数 | transition组数 | 能得出的结论 |
|---|---:|---:|---|
| 距冻结candidate ≤10m | 17 | 15 | 处于候选buffer邻域；不是自动成员 |
| 距离 >10m且≤20m | 14 | 12 | 处于candidate-pair距离范围；不满足自动归属条件 |
| 距离>20m，非shortcut直接邻接一个complex | 16 | 16 | 外部道路连接候选；邻接不等于属于该complex |
| 距离>20m，非shortcut邻接多个complex | 7 | 7 | 多complex之间的连接位置；不可任取一个归属 |
| 距离>20m，无非shortcut直接complex锚点 | 1 | 1 | 仅靠当前一跳证据不能归属 |
| 合计 | 55 | 51 | 全部保持未指派 |

最近candidate距离：最小4.87m，中位18.41m，最大258.50m。
10m内涉及多个complex的节点有1个，20m内有4个，进一步说明不能nearest-assign。
“>20m”只证明不在当前candidate的这个空间范围内，不证明现实中不属于某个
更大的交叉口，也不覆盖roundabout特殊规则。

## 原生邻接与控制证据

对每个目标及其精确transition aliases，枚举原生outgoing记录，用对向edge读取
incoming auto access。以双向transition组去重邻居，不用坐标折叠身份。
输出210条邻接记录，其中30条为native shortcut。主分类的complex邻接和分支度
均排除shortcut；全量记录仍保留并有显式flag，避免高层shortcut制造虚假邻接。

非shortcut邻居度分布（GraphId口径）：

| 邻居数 | 节点数 |
|---|---:|
| 2 | 6 |
| 3 | 30 |
| 4 | 18 |
| 5 | 1 |

49/55个GraphId、45/51个transition组有分支或native signal/roundabout证据。
其中native signal为5个GraphId/4组；本次相邻edge未发现native roundabout标记。
这些是层级关联合并后的局部auto拓扑证据，不等于重新运行过S2B candidate规则，
也不证明应新增45个intersection complexes。没有复核完整控制/法律restriction，
不能据此认证movement许可。

## 对最小接口修正的含义

1. 已有身份读取缺口确实存在，不能继续把这些节点解释成“原生图未知”。
2. 但大部分位置有分支证据，不能统一作为无控制的外部shape point直接放行。
3. 靠近旧complex的27个transition组需要明确成员/边界语义；远离旧candidate
   的24组需要区分道路连接位置与遗漏的独立junction。当前数据支持提出这些
   审查类别，不支持自动分配或自动认证。
4. 因此本轮未证明“保持所有membership不变，仅补88条edge端点即可修复391个
   encounters”。同样也没有证明必须全网重新聚类。

不再通过删除boundary或encounter绕过这些证据。此前391条UNKNOWN、117个
movement keys及两个断链记录均未被本工具改写。reverse、fallback、M3、profile、
41场景和论文结论全部保持不变。本轮到审查结果为止，不自动发起新的修复阶段。

## 交付

- [55节点关系表](native_node_complex_context_final/node_context.csv)
- [210条原生邻接及shortcut标记](native_node_complex_context_final/native_adjacency.csv)
- [汇总、输入哈希和资源统计](native_node_complex_context_final/summary.json)
- [QA](native_node_complex_context_final/qa.json)
- 可复现入口：`python -m stage4.tools.audit_native_node_complex_context`

公开表仅有静态网络身份和距离/拓扑统计，没有订单、车辆或轨迹坐标。
本地早期诊断草稿保留，最终产品为 `native_node_complex_context_final`。
运行约1.55秒、峰值工作集250.66MiB、CPU单进程、无GPU，无稠密距离矩阵，
无routing/matrix/inference调用。冻结节点、边、membership、complex、movement、
boundary与读取的原生tiles哈希前后相同。
