"""Run the dual-route weighted-RRF downstream pipeline on the 20 refusal samples.

No VLM gating. The production rules that CAN still refuse: no candidate, no
object_name, empty body evidence, plus the LLM's own "cannot confirm" wording.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/app")

from router.qa_rt import (
    _build_image_answer_question,
    _build_insufficient_text_answer,
    _build_visual_anchor,
)
from service.answer_generation_service import complete_answer
from service.icon_catalog_service import canonicalize_evidence
from service.retrieval_service import retrieve_visual_anchor_text
from utils.database import SessionLocal

ROUTE = "rrf_image_1_multimodal_0_6"
QUERY_TAIL = "含义 亮起 怎么办 处理方法"
REFUSAL_MARKERS = ("无法确认", "无法可靠", "证据不足", "无法从当前文档", "未检索到", "无法给出", "不足")


def detect_refusal(answer: str) -> bool:
    return any(marker in answer for marker in REFUSAL_MARKERS)


def main() -> None:
    cached_path, dataset_path, out_path = (Path(a) for a in sys.argv[1:4])
    cached = json.loads(cached_path.read_text(encoding="utf-8"))
    records = {r["id"]: r for r in (json.loads(l) for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip())}

    rows = [r for r in cached.get("image_rows") or [] if r.get("should_refuse")]
    db = SessionLocal()
    results = []
    try:
        for i, row in enumerate(rows, start=1):
            rec = records.get(row["id"]) or {}
            candidates = row.get("route_candidates", {}).get(ROUTE) or []
            question = rec.get("question") or "这张图片是什么，应该如何理解或使用？"
            status = "refused_no_candidate"
            answer = "当前手册未找到可靠视觉证据，无法给出确定结论。"
            anchor_info = None
            text_query = None
            text_evidence = []
            if not candidates:
                results.append({"id": row["id"], "negative_type": row.get("negative_type"),
                                "refusal_evaluable": row.get("refusal_evaluable"),
                                "status": status, "anchor": None, "answer": answer,
                                "llm_refusal": True, "text_query": None})
                print(f"[{i}/{len(rows)}] {row['id']} {status}", flush=True)
                continue
            anchor = candidates[0]
            object_name = str(anchor.get("object_name") or "").strip()
            text_query = " ".join(p for p in (object_name, QUERY_TAIL) if p) or question
            visual_anchor = _build_visual_anchor(anchor)
            visual_evidence = {**canonicalize_evidence(anchor), "evidence_role": "visual"}
            anchor_info = {"object_name": object_name, "page": anchor.get("page"),
                           "chunk_id": anchor.get("chunk_id"), "icon_group_id": anchor.get("icon_group_id"),
                           "rrf_score": anchor.get("rrf_score")}
            if not object_name:
                status = "refused_no_object_name"
                answer = _build_insufficient_text_answer(visual_evidence)
            else:
                text_evidence = retrieve_visual_anchor_text(db, rec.get("manual_id") or "demo_manual",
                                                            text_query, visual_anchor, top_k=4)[:4]
                if not text_evidence:
                    status = "refused_no_text_evidence"
                    answer = _build_insufficient_text_answer(visual_evidence)
                else:
                    evidence = [visual_evidence, *text_evidence]
                    vision = {"recognized_object": object_name, "visual_description": "",
                              "ocr_text": "", "candidate_names": [], "visual_keywords": [],
                              "normalized_query": text_query, "risk_level": "unknown"}
                    visual_match = {"object_name": visual_evidence.get("object_name"),
                                    "section_title": visual_evidence.get("section_title"),
                                    "page": visual_evidence.get("page"), "modality": visual_evidence.get("modality"),
                                    "icon_group_id": visual_evidence.get("icon_group_id")}
                    answer = complete_answer(_build_image_answer_question(vision, question, text_query, visual_match), evidence)
                    status = "answered"
            results.append({"id": row["id"], "negative_type": row.get("negative_type"),
                            "refusal_evaluable": row.get("refusal_evaluable"),
                            "status": status, "anchor": anchor_info,
                            "text_query": text_query,
                            "text_evidence": [{"chunk_id": e.get("chunk_id"), "page": e.get("page"),
                                               "section_title": e.get("section_title"),
                                               "content_text": str(e.get("content_text") or "")[:200]} for e in text_evidence],
                            "answer": answer, "llm_refusal": detect_refusal(answer)})
            print(f"[{i}/{len(rows)}] {row['id']} {status} anchor={object_name!r} llm_refusal={detect_refusal(answer)}", flush=True)
    finally:
        db.close()
    out_path.write_text(json.dumps({"count": len(results), "rows": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nDONE")


if __name__ == "__main__":
    main()
