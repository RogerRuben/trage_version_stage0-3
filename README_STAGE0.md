# 西安滴滴轨迹 Stage0

本目录包含一套可复现的小样本验证流程：流式抽样、GPS 清洗、OSM 路网下载、轻量 map matching、道路语义融合、行为特征和案例图。

## 已执行配置

- 数据：`10.1/gps_20161001`（无表头五列）
- 推断字段：`driver_id, order_id, timestamp, lon, lat`
- 时段：2016-10-01 17:00–19:00（Asia/Shanghai）
- 样本：500 个完整订单
- 坐标：原始值经诊断符合 GCJ-02；匹配前转换为 WGS84
- 路网：2026-07-05 下载的 OSM 西安固定主城区范围，驾车网络

## 复现命令

```powershell
python .\scripts\prepare_sample.py --input .\10.1\gps_20161001 --output-dir .\stage0_output --start-local "2016-10-01 17:00:00" --hours 2 --orders 500

python .\scripts\download_osm_network.py --region fixed-xian-core --output-dir .\stage0_output\network

python .\scripts\run_stage0.py --sample .\stage0_output\sample_raw.parquet --graphml .\stage0_output\network\xian_stage0_drive.graphml --output-dir .\stage0_output --input-crs gcj02
```

若 GraphML 已存在，路网导出可加 `--reuse-existing`，不会重新请求 OSM。

## 关键产物

- `stage0_output/stage0_report.md`：结论与核心指标
- `stage0_output/stage0_order_table.parquet`：Stage1 label 的订单级底表
- `stage0_output/matched_points.parquet`：清洗、匹配和道路语义后的点表
- `stage0_output/order_quality.csv`：订单清洗质量
- `stage0_output/map_matching_quality.csv`：订单匹配质量
- `stage0_output/trajectory_features.csv`：primitive indicators
- `stage0_output/figures/`：分布图和五类订单案例
- `stage0_output/network/xian_stage0_drive.graphml`：带拓扑的权威路网图
- `stage0_output/network/xian_stage0_edges.gpkg` 与 `xian_stage0_nodes.gpkg`：路段和节点语义

## 方法边界

本轮 matcher 是最近道路投影加拓扑/平行道路跳转审计，适合可行性筛查，不是生产级 HMM/Viterbi。正式 Stage1 前还需确认提供方坐标说明，换用 2016 同期路网，并加入候选道路、方向和网络转移概率。
