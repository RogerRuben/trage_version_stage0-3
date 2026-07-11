# Local topology-aware matcher with FMM-style gap repair

The production Stage0 matcher for the compact split experiment is:

```text
local_topology_fmm
= geometric projection
+ local topology audit
+ FMM-style routed gap repair
+ compact retention/pruning
```

This is not a full HMM/Viterbi matcher, and it should not be described as a standard FMM implementation. It is a lightweight topology-aware production matcher designed for the current Xi'an DiDi split pipeline.

## Why this matcher is used

The full HMM/Viterbi matcher was implemented and tested, but it was not selected as the production matcher because the lightweight local-topology FMM-style matcher achieved sufficient spatial accuracy, higher observed route-length coverage, lower fallback ratio, and much lower retained storage under the compact split workflow.

## Output interpretation

The matcher produces a continuous route representation by retaining observed traversals and inserting routed-but-unobserved links when local topology repair is needed.

```text
observed traversal
  GPS evidence exists on the link and can support realized behavior labels.

inferred_path traversal
  The link is inserted for route continuity. It is useful for route geometry and topology, but it does not carry direct realized driving behavior.
```

Therefore inferred links are retained for route continuity but excluded from realized LCS/IIS/RTS/PMIS label construction. They may still support static or contextual route features, such as GNS, route position, link density, or geometry continuity.

## Recommended paper wording

English:

> We use a local topology-aware map matcher with FMM-style gap repair. The matcher combines geometric road projection, local topology auditing, and routed repair for short path discontinuities. It is not a full HMM/Viterbi matcher. Routed-but-unobserved links are retained for route continuity but excluded from realized behavior label construction.

Chinese:

> 本研究采用局部拓扑感知地图匹配器，并带有类 FMM 的路径断裂修复机制。该方法结合几何投影、局部拓扑审计和短距离路径断裂修复，但不是完整的 HMM/Viterbi 地图匹配器。由路径修复插入、但缺少 GPS 行为观测的 inferred links 仅用于路线连续性和静态结构特征，不直接用于 LCS/IIS/RTS/PMIS 等 realized behavior label 构造。
