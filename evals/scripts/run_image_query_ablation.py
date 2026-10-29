"""A/B: query construction — object_name-only (A) vs production VLM-heavy (B).

Anchors are fixed to the cached weighted-RRF Top1 (rrf_image_1_multimodal_0_6).
Only the body-retrieval query changes. The answer prompt is held constant
(minimal vision, same as run A) so any difference is attributable to retrieval.

query A (already run): object_name + "含义 亮起 怎么办 处理方法"
query B (this run):    production _build_image_text_query(vision, question, anchor),
                       re-running first-pass VLM understanding for real vision fields.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/app")

from router.qa_rt import (
    _build_image_answer_question,
    _build_image_text_query,
    _build_insufficient_text_answer,
    _build_visual_anchor,
)
from service.answer_generation_service import complete_answer
from service.icon_catalog_service import canonicalize_evidence
from service.retrieval_service import retrieve_visual_anchor_text
from service.vision_understanding_service import understand_vehicle_image
from utils.database import SessionLocal

ROUTE = "rrf_image_1_multimodal_0_6"
QUERY_TAIL = "含义 亮起 怎么办 处理方法"


def anchor_matches(row: dict, cand: dict) -> bool:
    # Every visual object owns exactly one chunk, so icon and operation hits
    # are judged by the same audited chunk ID.
    return str(cand.get("chunk_id")) in set(row.get("expected_visual_chunk_ids") or [])


def minimal_vision(text_query: str, title: str) -> dict:
    return {
        "recognized_object": title or "unknown",
        "visual_description": "",
        "ocr_text": "",
        "candidate_names": [],
        "visual_keywords": [],
        "normalized_query": text_query,
        "risk_level": "unknown",
    }


def main() -> None:
    cached_path, dataset_path, img_root, out_path = (Path(a) for a in sys.argv[1:5])
    cached = json.loads(cached_path.read_text(encoding="utf-8"))
    records = {r["id"]: r for r in (json.loads(l) for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip())}
    rows = [r for r in cached.get("image_rows") or [] if not r.get("should_refuse")]

    db = SessionLocal()
    results = []
    vlm_fail = 0
    try:
        for i, row in enumerate(rows, start=1):
            rec = records.get(row["id"]) or {}
            candidates = row.get("route_candidates", {}).get(ROUTE) or []
            if not candidates:
                results.append({"id": row["id"], "status": "no_anchor", "error": "missing route candidates"})
                continue
            anchor = candidates[0]
            anchor_correct = anchor_matches(row, anchor)
            visual_anchor = _build_visual_anchor(anchor)
            visual_evidence = {**canonicalize_evidence(anchor), "evidence_role": "visual"}
            question = rec.get("question") or "这张图片是什么，应该如何理解或使用？"

            # re-run first-pass VLM for real vision fields
            img_rel = str(rec.get("image_path") or "")
            img_path = img_root / img_rel if img_rel else None
            vision = {}
            if img_path is not None and img_path.exists():
                vision = understand_vehicle_image(img_path, question) or {}
                if vision.get("vision_error"):
                    vlm_fail += 1
            else:
                vlm_fail += 1

            query_b = _build_image_text_query(vision, question, anchor)
            text_evidence = retrieve_visual_anchor_text(db, rec.get("manual_id") or "demo_manual", query_b, visual_anchor, top_k=4)[:4]
            evidence = [visual_evidence, *text_evidence]

            if not text_evidence:
                answer = _build_insufficient_text_answer(visual_evidence)
                status = "insufficient_text_evidence"
            else:
                title = visual_evidence.get("object_name") or visual_evidence.get("section_title") or "unknown"
                vis = minimal_vision(query_b, title)
                visual_match = {"object_name": visual_evidence.get("object_name"),
                                "section_title": visual_evidence.get("section_title"),
                                "page": visual_evidence.get("page"),
                                "modality": visual_evidence.get("modality"),
                                "icon_group_id": visual_evidence.get("icon_group_id")}
                answer = complete_answer(_build_image_answer_question(vis, question, query_b, visual_match), evidence)
                status = "verified"

            results.append({
                "id": row["id"], "anchor_correct": anchor_correct, "status": status,
                "query": query_b,
                "vision": {k: vision.get(k) for k in ("visual_description", "ocr_text", "candidate_names", "visual_keywords", "normalized_query")},
                "text_evidence": [{"chunk_id": e.get("chunk_id"), "page": e.get("page"),
                                   "section_title": e.get("section_title"),
                                   "content_text": str(e.get("content_text") or "")[:400],
                                   "match_method": e.get("match_method")} for e in text_evidence],
                "answer": answer,
            })
            print(f"[{i}/{len(rows)}] {row['id']} anchor={anchor_correct} status={status} evid={len(text_evidence)}", flush=True)
    finally:
        db.close()
    summary = {"count": len(rows), "vlm_fail": vlm_fail,
               "verified": sum(r.get("status") == "verified" for r in results),
               "insufficient_text_evidence": sum(r.get("status") == "insufficient_text_evidence" for r in results)}
    out_path.write_text(json.dumps({"run": {"mode": "query_ablation_B_vlm", "route": ROUTE, "answer_prompt": "minimal_vision(constant)"}, "summary": summary, "image_rows": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nSUMMARY", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
