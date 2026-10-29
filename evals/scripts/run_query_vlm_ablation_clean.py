"""Clean 2-variant: VLM off vs VLM on, holding query template minimal.

Variant 1 (no VLM): query = question + object_name + section_title ; prompt = no VLM fields
Variant 2 (VLM):    query = variant1 + normalized_query + visual_description ; prompt = adds VLM初诊描述

Same anchors (cached rrf_image_1_multimodal_0_6 Top1), same body-retrieval service.
Vision fields are reused from the earlier VLM re-run (image_queryB_vlm_20260807.json).
"""

from __future__ import annotations

import json
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


def anchor_matches(row: dict, cand: dict) -> bool:
    # Every visual object owns exactly one chunk, so icon and operation hits
    # are judged by the same audited chunk ID.
    return str(cand.get("chunk_id")) in set(row.get("expected_visual_chunk_ids") or [])


def minimal_vision(text_query: str, title: str) -> dict:
    return {"recognized_object": title or "unknown", "visual_description": "", "ocr_text": "",
            "candidate_names": [], "visual_keywords": [], "normalized_query": text_query, "risk_level": "unknown"}


def real_vision(text_query: str, title: str, vision: dict) -> dict:
    return {"recognized_object": title or "unknown",
            "visual_description": str(vision.get("visual_description") or ""),
            "ocr_text": str(vision.get("ocr_text") or ""),
            "candidate_names": vision.get("candidate_names") or [],
            "visual_keywords": vision.get("visual_keywords") or [],
            "normalized_query": text_query, "risk_level": "unknown"}


def generate(db, manual_id, query, visual_anchor, visual_evidence, question, vision, evidence):
    if not evidence:
        return _build_insufficient_text_answer(visual_evidence), "insufficient_text_evidence"
    title = visual_evidence.get("object_name") or visual_evidence.get("section_title") or "unknown"
    visual_match = {"object_name": visual_evidence.get("object_name"),
                    "section_title": visual_evidence.get("section_title"),
                    "page": visual_evidence.get("page"), "modality": visual_evidence.get("modality"),
                    "icon_group_id": visual_evidence.get("icon_group_id")}
    answer = complete_answer(_build_image_answer_question(vision, question, query, visual_match), evidence)
    return answer, "verified"


def main() -> None:
    cached_path, dataset_path, vision_path, out_path = (Path(a) for a in sys.argv[1:5])
    cached = json.loads(cached_path.read_text(encoding="utf-8"))
    records = {r["id"]: r for r in (json.loads(l) for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip())}
    vlm = {r["id"]: r.get("vision") or {} for r in json.loads(vision_path.read_text(encoding="utf-8")).get("image_rows") or []}
    rows = [r for r in cached.get("image_rows") or [] if not r.get("should_refuse")]

    db = SessionLocal()
    results = []
    try:
        for i, row in enumerate(rows, start=1):
            rec = records.get(row["id"]) or {}
            candidates = row.get("route_candidates", {}).get(ROUTE) or []
            if not candidates:
                results.append({"id": row["id"], "status": "no_anchor"})
                continue
            anchor = candidates[0]
            object_name = str(anchor.get("object_name") or "").strip()
            section_title = str(anchor.get("section_title") or "").strip()
            question = rec.get("question") or "这张图片是什么，应该如何理解或使用？"
            visual_anchor = _build_visual_anchor(anchor)
            visual_evidence = {**canonicalize_evidence(anchor), "evidence_role": "visual"}
            vis = vlm.get(row["id"]) or {}

            query_off = " ".join(p for p in (question, object_name, section_title) if p) or question
            query_vlm = " ".join(p for p in (query_off, str(vis.get("normalized_query") or ""), str(vis.get("visual_description") or "")) if p) or query_off

            # variant 1: no VLM query, no VLM prompt
            ev_off = retrieve_visual_anchor_text(db, rec.get("manual_id") or "demo_manual", query_off, visual_anchor, top_k=4)[:4]
            ans_off, st_off = generate(db, rec.get("manual_id") or "demo_manual", query_off, visual_anchor, visual_evidence, question,
                                       minimal_vision(query_off, visual_evidence.get("object_name") or object_name), [visual_evidence, *ev_off])
            # variant 2: VLM query + VLM prompt
            ev_vlm = retrieve_visual_anchor_text(db, rec.get("manual_id") or "demo_manual", query_vlm, visual_anchor, top_k=4)[:4]
            ans_vlm, st_vlm = generate(db, rec.get("manual_id") or "demo_manual", query_vlm, visual_anchor, visual_evidence, question,
                                       real_vision(query_vlm, visual_evidence.get("object_name") or object_name, vis), [visual_evidence, *ev_vlm])

            results.append({
                "id": row["id"], "anchor_correct": anchor_matches(row, anchor),
                "query_off": query_off, "query_vlm": query_vlm,
                "variant1": {"query": query_off, "prompt_vlm": False, "evidence": [{"chunk_id": e.get("chunk_id"), "page": e.get("page"), "content_text": str(e.get("content_text") or "")[:300]} for e in ev_off], "answer": ans_off, "status": st_off},
                "variant2": {"query": query_vlm, "prompt_vlm": True, "evidence": [{"chunk_id": e.get("chunk_id"), "page": e.get("page"), "content_text": str(e.get("content_text") or "")[:300]} for e in ev_vlm], "answer": ans_vlm, "status": st_vlm},
            })
            print(f"[{i}/{len(rows)}] {row['id']} v1={st_off}/{len(ev_off)} v2={st_vlm}/{len(ev_vlm)}", flush=True)
    finally:
        db.close()
    out_path.write_text(json.dumps({"count": len(rows), "image_rows": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nDONE")


if __name__ == "__main__":
    main()
