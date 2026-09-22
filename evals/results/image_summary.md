# 图片问答评测 Summary

## 数据与口径

- 70 条可回答视觉样本：50 条 icon、20 条 operation figure。
- icon 按 `icon_group_id` 评分，operation figure 按精确 `visual_chunk_id` 评分。
- 图片端到端正文 Gold 中，69 条包含可评分的 `text_evidence_units`。

## Group–Occurrence 索引

| Icon occurrence | Icon group | 候选身份减少 |
|---:|---:|---:|
| 377 | 164 | **56.5%** |

## 视觉召回

| 路线 | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| 白底图片向量 | 92.86% | 100.00% | 100.00% | 0.9548 |
| 多模态向量 | 88.57% | 98.57% | 98.57% | 0.9300 |
| 图片 1.0 / 多模态 1.0 RRF | **95.71%** | 98.57% | **100.00%** | 0.9743 |
| 图片 1.0 / 多模态 0.6 RRF | **95.71%** | 98.57% | **100.00%** | **0.9750** |
| RRF + OpenCV 0.2 | 77.14% | 97.14% | 100.00% | 0.8755 |
| 旧图片硬门控 | 84.29% | 94.29% | 97.14% | 0.8993 |

双路 RRF 将 70 条混合视觉样本 Hit@1 从 92.86% 提升到 95.71%。50 条 icon-only 集合中，白底图片向量 Group Hit@1 为 94%，等权 RRF 为 98%。

候选改选式 VLM 二验将锚点准确率从 95.71% 降到 91.43%，因此默认关闭；OpenCV 只保留为离线诊断。

## 图片端到端

端到端补测使用图片 1.0 / 多模态 0.6 的缓存 Top1 锚点，再运行 occurrence 页面扩展、正文检索、重排、AutoMerge 和生成。

| 指标 | 结果 |
|---|---:|
| 锚点正确率 | 95.71% |
| Complete Unit Hit@4 | **86.96%** |
| Macro Unit Recall@4 | 89.13% |
| Answer Correctness | **93.14%** |
| Groundedness | **98.43%** |

简历按一位小数表示为：视觉 Hit@1 92.8% → 95.7%，必答点完整覆盖 87%，Answer Correctness 93.1%，Groundedness 98.4%。图片端结果由独立 Claude Judge 配合人工逐条复核。

