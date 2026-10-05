# 项目磁盘清理 · 2026-10-05

本轮只清理可确认的中间副本、可再生成缓存和干净临时源码，不运行采集或全天实验，不删除模型、主数据和冻结结果，不修改文件ACL。

## 实际结果

| 项目 | 占用/释放 |
|---|---:|
| 删除原副本及再生成文件 | 3.710 GiB |
| 保留可恢复快照ZIP | 0.528 GiB |
| 净释放磁盘空间 | 3.182 GiB |
| D盘可用空间 | 约10.69 → 13.89 GiB |

已处理：35个旧交叉口SQLite/观测JSON/Parquet/查询清单大副本（3.384 GiB），705个Python字节码或测试缓存目录（0.118 GiB），无本地改动/分支/stash的上游Valhalla 3.8.2临时源码（0.207 GiB）。不是释放同等数量的RAM或显存。

## 保留与校验

- 完整信号并集699处、地图/高德支持集合693处未变；83个封存交付文件及其外部保护来源，共234个文件的SHA校验通过。
- `map_data/signal_gapfill_20261004/`、旧信号最终版本、当前观测状态、原始API证据、地图、来源映射和请求预算账本保留。
- 原始RAR/OSM ZIP、Stage1输入、Stage2历史store/主路线数据/transfer shards、模型与预测缓存、Stage3 CDF/profile、Stage4冻结结果与原第二组未完成产物保留。
- 主目录原先已有的大量Git删除/修改没有恢复、提交或扩大。维护清单只在当前工作树提交。

磁盘扫描遇到权限受限目录，包括部分Stage1模型、Stage1 output_v3和pytest缓存。没有改ACL或强制读取/删除。初步33.12 GiB是当时**可读取文件**统计，不代表包含所有受限数据的项目完整占用。

## 恢复方式

35个大快照删除前已写入并逐文件核验长度和SHA，归档保留完整项目相对路径：

[快照ZIP](D:/pycodes/didi_xian_raw/map_data/_cleanup_archives/snapshots_20261005.zip)

ZIP SHA256：`b3ece8b4d9e4a8e2747c765b43a7b738025474cfac9a723ac116219572c0a55e`。

如果要重新运行引用旧before/after快照的历史审计，先校验ZIP，再在项目根目录解压恢复对应路径；不要覆盖后来重新生成的同名文件。当前最终成果和推理输入不需要恢复这些回滚副本。

Python缓存会在后续import/测试时自动再生成。临时源码可从`https://github.com/valhalla/valhalla.git`恢复并checkout `17af0d031babb05d24304a16e89239a074c7bc8c`；安装好的Valhalla绑定和tiles没有删除。

## 记录与边界

[详细清单](project_disk_cleanup_20261005.json)记录所有路径、字节数、恢复类别和跳过项。脚本：[cleanup_unused_project_files.ps1](../../scripts/maintenance/cleanup_unused_project_files.ps1)。

第一次执行在快照已核验、原副本已删除后，因访问受限pytest目录而停止。已修复为跳过受限目录、不改ACL，并从既有归档恢复操作清单完成后续清理；没有重建或覆盖归档。199条扫描/跳过记录包含受限数据目录，不能将它们都称为无用缓存。

C盘不在本轮删除范围内；系统目录、Codex附件、下载资料和其他项目没有清理。没有为了腾空间而删除仍被当前代码或冻结协议引用的大型研究数据。
