"""Run the 15-question gold set through the real pipeline and write a report."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from llm_client import MockLLMClient, call_model
from models import Candidate, Chunk
from pipeline import ingest_documents, run_pipeline
from scoring import score_chunks
from topk_heap import select_top_k

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
GOLD_SET = ROOT / "tests" / "gold_set.md"
OUTPUT = ROOT / "evaluation_results.md"
SOURCE_ID_PATTERN = re.compile(r"\bFB-\d{3}\b")
CHUNK_BLOCK_START = "<BEGIN_UNTRUSTED_SELECTED_CHUNKS>\n"
CHUNK_BLOCK_END = "\n<END_UNTRUSTED_SELECTED_CHUNKS>"


def _read_gold_set() -> list[dict[str, Any]]:
    cases = []
    for line in GOLD_SET.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or not line.split("|")[1].strip().isdigit():
            continue
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        if len(columns) != 4:
            raise ValueError(f"unexpected gold-set row: {line}")
        cases.append(
            {
                "number": int(columns[0]),
                "question": columns[1],
                "expected": columns[2],
                "source_ids": SOURCE_ID_PATTERN.findall(columns[3]),
            }
        )
    return cases


def _selected_for_query(query: str, chunks: list[Chunk]) -> list[Candidate]:
    scored = score_chunks(query, chunks)
    selected_rows = select_top_k(
        [
            (
                candidate.chunk.doc_id,
                candidate.chunk.chunk_id,
                candidate.chunk.section,
                candidate.score,
            )
            for candidate in scored
        ],
        5,
    )
    by_id = {
        (candidate.chunk.doc_id, candidate.chunk.chunk_id): candidate
        for candidate in scored
    }
    return [by_id[(doc_id, chunk_id)] for doc_id, chunk_id, _, _ in selected_rows]


def _status_for_case(case: dict[str, Any]) -> str:
    expected = case["expected"].casefold()
    if "conflict" in expected or "conflicting" in expected:
        return "conflict"
    if not case["source_ids"] or "not provided" in expected:
        return "insufficient_evidence"
    return "ok"


def _mock_answer(case: dict[str, Any], selected: list[Candidate]) -> str:
    status = _status_for_case(case)
    if status != "ok":
        return json.dumps({"status": status, "action_items": []}, ensure_ascii=False)

    citations = []
    for source_id in case["source_ids"]:
        match = next(
            (
                candidate.chunk
                for candidate in selected
                if source_id in candidate.chunk.text
            ),
            None,
        )
        if match is not None:
            citations.append(
                {
                    "doc_id": match.doc_id,
                    "chunk_id": match.chunk_id,
                    "excerpt": source_id,
                }
            )
    if not citations and selected:
        chunk = selected[0].chunk
        citations.append(
            {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "excerpt": chunk.text[: min(80, len(chunk.text))],
            }
        )

    item = {
        "category": "citizen feedback",
        "location": "as stated in the cited feedback",
        "priority": "review",
        "suggested_department": "relevant department",
        "summary": case["expected"][:500],
        "citations": citations,
    }
    if not citations:
        return json.dumps({"status": "insufficient_evidence", "action_items": []})
    return json.dumps({"status": "ok", "action_items": [item]}, ensure_ascii=False)


def _parse_selected(prompt: str) -> list[Candidate]:
    start = prompt.index(CHUNK_BLOCK_START) + len(CHUNK_BLOCK_START)
    end = prompt.index(CHUNK_BLOCK_END, start)
    records = json.loads(prompt[start:end])
    return [
        Candidate(
            chunk=Chunk(
                doc_id=record["doc_id"],
                chunk_id=record["chunk_id"],
                section=record["section"],
                text=record["text"],
            ),
            score=0.0,
        )
        for record in records
    ]


def _render_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    cases = _read_gold_set()
    data_files = sorted(
        path
        for path in DATA_DIR.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".txt", ".md"}
    )
    if not data_files:
        raise RuntimeError(f"No .txt or .md evaluation data found in {DATA_DIR}")
    ingest_documents(data_files)
    from ingest import ingest_files

    all_chunks = ingest_files(data_files)

    configured = all(
        os.environ.get(name)
        for name in ("RAG_LLM_ENDPOINT", "RAG_LLM_MODEL", "RAG_LLM_API_KEY")
    )
    mode = "configured model endpoint" if configured else "mock client"
    rows = []
    supported_count = 0

    for case in cases:
        predicted_selection = _selected_for_query(case["question"], all_chunks)
        captured_prompt: list[str] = []
        if configured:
            def invoke(prompt: str, *, system_prompt: str) -> str:
                captured_prompt.append(prompt)
                return call_model(prompt, system_prompt=system_prompt)
        else:
            mock_client = MockLLMClient([_mock_answer(case, predicted_selection)])

            def invoke(prompt: str, *, system_prompt: str) -> str:
                captured_prompt.append(prompt)
                return mock_client(prompt, system_prompt=system_prompt)

        result = run_pipeline(case["question"], llm=invoke)
        selected = _parse_selected(captured_prompt[0]) if captured_prompt else []
        retrieved_ids = sorted(
            {
                source_id
                for candidate in selected
                for source_id in SOURCE_ID_PATTERN.findall(candidate.chunk.text)
            }
        )
        cited_ids = sorted(
            {
                source_id
                for item in result.get("action_items", [])
                for citation in item.get("citations", [])
                for source_id in SOURCE_ID_PATTERN.findall(citation["excerpt"])
            }
        )

        issues: list[str] = []
        expected_ids = set(case["source_ids"])
        if result["status"] == "ok":
            cited_items = result.get("action_items", [])
            if not cited_items:
                issues.append("ok status has no action item")
            if any(source_id not in expected_ids for source_id in cited_ids):
                issues.append("citation source differs from gold references")
            if any(source_id not in retrieved_ids for source_id in cited_ids):
                issues.append("citation source was not retrieved")
        if case["source_ids"] and not expected_ids.issubset(set(retrieved_ids)):
            issues.append("one or more expected source IDs were not retrieved")

        if expected_ids:
            supported = (
                result["status"] in {"ok", "conflict", "insufficient_evidence"}
                and expected_ids.issubset(set(retrieved_ids))
                and not issues
            )
        else:
            supported = (
                result["status"] == "insufficient_evidence"
                and not result.get("action_items")
                and not issues
            )
        if supported:
            supported_count += 1
        rows.append(
            "| {question} | {expected} | {retrieved} | {status} | {supported} | {issues} |".format(
                question=_render_cell(f'{case["number"]}. {case["question"]}'),
                expected=_render_cell(", ".join(case["source_ids"]) or "None"),
                retrieved=_render_cell(", ".join(retrieved_ids) or "None"),
                status=result["status"],
                supported="yes" if supported else "no",
                issues=_render_cell("; ".join(issues) or "None detected by source/citation checks"),
            )
        )

    report = [
        "# Gold-set evaluation",
        "",
        f"Run mode: **{mode}**.",
        "",
        "Support is assessed by expected-source retrieval, pipeline status, and citation/source checks. "
        "This is not a semantic entailment score; model judgments in real-endpoint mode require human review.",
        "",
        "| Question | Expected source IDs | Retrieved IDs | Status | Supported | Unsupported claims or wrong citations |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        f"**Support rate:** {supported_count}/{len(cases)} ({supported_count / len(cases):.1%}).",
        "",
    ]
    OUTPUT.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} using {mode}: {supported_count}/{len(cases)} supported")


if __name__ == "__main__":
    main()
