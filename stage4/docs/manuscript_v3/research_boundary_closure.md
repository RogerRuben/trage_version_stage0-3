# 接受冻结接口边界，收敛论文

依据：用户明确选择“接受边界、收敛论文”。本轮为文档修订，不是新实验。

## 停止线

- 不追加端点/complex审计或gate组合消融。
- 不修Stage3接口，不重新聚类，不变更membership、reverse、fallback或profile。
- 不重训/推断M3，不重跑41场景，不启动Stage5或common evaluator。
- 不把D4 oracle收益写成修复收益，不把保留旧结果写成接口已通过无误认证。

若未来重新打开修复，需明确的新授权；历史报告里的“下一步建议”不是授权。

## 论文现在回答的问题

在给定能力假设、冻结网络/路线接口、证据准入政策与时间设定下，名义车时如何
转化为兼容图、瞬时有效匹配容量与全天服务？

不是：真实AV技术固有地降低服务能力；也不是：修好接口后结论必然成立。
已知movement输入域不一致是实测局限，不能只泛称“数据可能有误”。

## 本轮修改

- 原位修订V3，Git保留历史，不创建重复母稿。
- Abstract、方法、Experimental Design、Results 6.5、Discussion、Limitations和Conclusion写明接口边界。
- Appendix G汇总既有D13及审计，区分诊断绕过和可部署修复。
- q_A称baseline-normalized AV active-hour level；瞬时匹配数不冒充日容量。
- figure plan、claims、storyline及研究README同步状态。
- 既有结果表和公式不变；没有新文献检索、新定理或结果生成。

## 剩余写作工作，不是新增研究阶段

1. 用既有数据完成5张主图，caption带冻结接口/时间限定。
2. 整合已核验closest literature，未完成的引用核验如实标记。
3. 整体论证与语言审校：因果措辞、分母、AV子图与mixed-system区分。
4. 最后选择期刊定位和格式；不以投稿目标为理由自动扩展实验或虚构新颖性。

当前仍是工作稿，最终图未绘制，投稿准备并未全部完成。

## 文档核验

对比修订前HEAD：主稿所有Markdown结果表逐行一致，所有display equations去空白后
一致；修订文件中的25个本地Markdown链接均存在；章节顺序及diff check通过。
本轮仅修改Markdown，没有执行新实验或写入模型/数据产品。
