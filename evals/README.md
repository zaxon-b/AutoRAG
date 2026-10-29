# Evaluation assets

本目录保存 AutoRAG 当前公开评测材料。

```text
evals/
├── datasets/
│   ├── owner_manual_text_eval_v4.jsonl
│   ├── owner_manual_image_eval_native_v3.jsonl
│   └── owner_manual_multimodal/images/
├── scripts/
│   └── 评测、Judge 与汇总报告生成代码
└── results/
    ├── text_summary.md
    └── image_summary.md
```

## 数据集

- 文本集：60 条，包含 50 条可回答问题和 10 条拒答问题；Gold 包含叶子等价组与原子必答点。
- 图片集：90 条，包含 70 条可回答样本与 20 条拒答审计样本；icon 按 `icon_group_id` 评价，operation figure 按精确视觉 Chunk 评价。
- 图片样本：JSONL 中引用的图片位于 `owner_manual_multimodal/images/`。

Gold Chunk/Group ID 对应本项目评测时构建的 `demo_manual` 索引。由于原始汽车手册 PDF 不随仓库分发，公开数据主要用于检查标注结构、评测代码与统计方法；严格复现需使用同一手册版本重新建库并映射 Gold ID。

数据中的手册摘录、截图和实拍图片仅作为研究评测样本；其内容权利仍归原始权利人所有，不随项目代码授予再分发许可。公开或二次使用前请确认相应授权。

## 公开结果

仓库只提交最终 summary，不提交逐题 audit、模型原始响应、判断缓存或临时 checkpoint。统一结果说明见仓库根目录 [Eval.md](../Eval.md)。

## 运行入口

主入口为 `backend/app/evaluate_owner_manual.py`：

```bash
cd backend
docker compose exec -T autorag_api python -m evaluate_owner_manual --mode all
docker compose exec -T autorag_api python -m evaluate_owner_manual --mode text --skip-generation
docker compose exec -T autorag_api python -m evaluate_owner_manual --mode image --limit-image 4
```

新运行产生的 JSON、逐题 audit 和 checkpoint 默认写入 `evals/results/runs/`，该目录不应作为公开结果提交。
