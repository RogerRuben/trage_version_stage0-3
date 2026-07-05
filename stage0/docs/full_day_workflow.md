# 西安滴滴轨迹全日 Stage0

## 已完成内容

1. 阅读 `map_data/osm-data-in-gis-formats-free.pdf` 并核对 Shapefile schema。
2. 从 2017-01-01 Geofabrik 中国路网中提取西安保守市域包络和轨迹核心区。
3. 将 28,334,617 个轨迹点按订单哈希到 128 个外存桶。
4. 完成全日 GPS 清洗、道路匹配、拓扑审计、道路语义融合和行为特征提取。
5. 输出订单底表、点级匹配分区、路线序列、道路暴露、质量报告和五类案例图。

## 坐标说明

输入五列按 `driver_id, order_id, timestamp, lon, lat` 解释。虽然提供的说明称坐标为 WGS84，但在同期 2017 OSM 路网上：

- 原值直接作为 WGS84：P90 匹配距离 120.17 m；
- 按 GCJ-02 转 WGS84：P90 匹配距离 10.78 m。

因此本次全日运行使用 GCJ-02→WGS84，同时在点表中保留 `source_lon/source_lat`。正式发表前仍应向数据提供方确认。

## 路网产物

- `map_data/xian_2017/xian_2017_envelope_*`：西安市域保守矩形包络；
- `map_data/xian_2017/xian_2017_core_*`：完整覆盖轨迹的匹配路网；
- `map_data/xian_2017/xian_2017_network_metadata.json`：来源、边界、字段和数量。

2017 免费包没有行政边界图层，因此 envelope 不是严格行政裁剪；也没有 `lanes` 字段，`maxspeed` 的有效标注约 1%。

## 全日关键产物

- `full_day_output/full_day_report.md`：全日结论；
- `full_day_output/full_day_stage0_orders.parquet`：119,018 个订单的 Stage1 基础表；
- `full_day_output/matched_points/`：128 个点级匹配分区；
- `full_day_output/route_parts/`：订单匹配路段序列；
- `full_day_output/road_exposure_parts/`：订单-道路等级暴露长表；
- `full_day_output/full_day_visual_case_index.csv`：五类案例索引；
- `full_day_output/coordinate_diagnostic_2017.json`：坐标诊断证据。

## 复现

```powershell
python .\stage0\scripts\extract_xian_2017_network.py --source-dir .\map_data\china-170101-free.shp --output-dir .\map_data\xian_2017

python .\stage0\scripts\run_full_day_2017.py --input .\10.1\gps_20161001 --roads .\map_data\xian_2017\xian_2017_core_roads.parquet --nodes .\map_data\xian_2017\xian_2017_core_nodes.parquet --output-dir .\full_day_output --input-crs gcj02

python .\stage0\scripts\generate_full_day_cases.py --roads .\map_data\xian_2017\xian_2017_core_roads.parquet --output-dir .\full_day_output
```

脚本会复用已完成的分桶 manifest 和各桶结果，可断点续跑。当前 matcher 是道路候选加密、精确线投影和拓扑审计，仍不是完整 HMM/Viterbi。
