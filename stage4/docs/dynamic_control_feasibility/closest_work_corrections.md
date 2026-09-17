# Closest-work corrections

日期：2026-09-14。只核对影响本轮模型定位的四篇已有文献，不是新增泛化综述。检索摘要是线索，不替代原文；以下明确区分可核准内容与尚未核准的强结论。

## 1. Akçay：灵活资源不等于决策期内循环释放

原文标识：[DOI 10.1111/j.1937-5956.2009.01095.x](https://doi.org/10.1111/j.1937-5956.2009.01095.x)。本轮出版社直接访问失败；可检索的[作者稿副本](https://citeseerx.ist.psu.edu/document?doi=ea8adbf2d8ba9113c1f87ae069476430120af241&repid=rep1&type=pdf)中，Section 2.1 的模型说明是在 service date 前的各期决定接受哪些任务、分配何种资源；培训机构例子包含课程接受与教师分配，也讨论延迟分配。

这支持“异质灵活资源的跨期保留/分配”作为近邻，不支持直接写成“服务完成后在同一决策期内返回资源池的动态车辆系统”。任务到达、接受和资源指派具有多期结构，与车辆执行行程后在新空间位置释放，是不同机制。

修正：删除把该文已确认为 reusable moving fleet 的说法。其是否有其他扩展精确包含本项目的空间释放、相关定理的全部假设，保持 **UNRESOLVED**；本轮没有完整核验全文所有变体，不能借此声称我们的空间模型必然新颖。

## 2. Shah：ride-pooling ILP 不等于当前非拼车匹配

[Neural Approximate Dynamic Programming for On-Demand Ride-Pooling，AAAI 原文页](https://ojs.aaai.org/index.php/AAAI/article/view/5388)讨论组合 ride-pooling action、ILP 与用于价值更新的 LP dual 难点。其建模背景是多乘客组合，不是普通一车一单的二部匹配。

本项目的基础一车一单约束具有经典二部匹配结构；不加额外耦合条件时，其 LP 松弛存在整数最优解。加入 Gamma 等侧约束后的性质应另行分析。生产求解器使用 MILP 并不等于已经出现该文的 pooling 松弛问题。

修正：不能用该文推出“当前所有 LP dual 都无效”。未来时空流近似的对偶价、当前 assignment 松弛的对偶价和离散动态价值也不是同一对象。本轮不计算任何一种 dual，不宣称其近似精度或算法保证。

## 3. Xu：当前收益加未来车辆价值已有直接先例

[Large-Scale Order Dispatch in On-Demand Ride-Hailing Platforms，ACM 出版记录](https://doi.org/10.1145/3219819.3219824)的摘要明确把历史学习的时空车辆价值用于实时组合派单，并同时考虑即时与未来收益。

因此，“派单时考虑未来车辆价值”本身不能作为新贡献。我们的有限反例只说明一个明确遗漏兼容分布的表示不充分，并不证明这类 learning-and-planning 框架不能增加兼容状态。是否优于同信息简单规则、是否需要超出常规状态扩展的方法，必须独立验证；当前两项均未得到复杂方法必要性的证据。

## 4. Talluri–van Ryzin：机会成本有依据，保证不能搬用

[An Analysis of Bid-Price Controls for Network Revenue Management，Management Science 出版摘要](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.44.11.1577?journalCode=mnsc)说明 bid-price control 一般不最优，并在适当大容量条件和正确价格下给出渐近结果。

该结论可以帮助解释资源机会成本，但不能直接转移成有限异质车辆、空间释放、session、deadline 和 Gamma 下的最优性保证。本文两动作差值只是有限期决策的标准比较；不是新 bid-price theorem。Hall deficiency 则描述给定当前图的匹配缺口，不能直接当作动态 shadow price。

## 5. 对本项目定位的净影响

保留：兼容分布的空间构成可影响全车队 post-decision value；即时匹配能力与动态机会成本需要分开。

撤回或不建立：ADP 无法表达兼容性、ILP 必然导致无用 dual、灵活资源文献已经覆盖/绝不覆盖循环空间释放、新复杂控制器必要、真实派单收益成立。

本轮没有新增方法优越性主张。文献核验的访问深度不足以支持的内容已保留 unresolved，不以二手检索摘要补写保证。主报告的有限模型证明不依赖这些未核准的保证。
