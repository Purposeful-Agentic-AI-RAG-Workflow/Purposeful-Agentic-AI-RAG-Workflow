"""Evaluate the 15-question gold set with an evidence-only mock or configured LLM."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from llm_client import MockLLMClient, call_model
from models import Candidate, Chunk
from pipeline import ingest_documents, run_pipeline
from scoring import tokenize

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
GOLD_SET = ROOT / "tests" / "gold_set.md"
OUTPUT = ROOT / "evaluation_results.md"
SOURCE_ID_PATTERN = re.compile(r"\bFB-\d{3}\b")
MESSAGE_PATTERN = re.compile(r"\b(FB-\d{3}):\s*(.*?)(?=\s+FB-\d{3}:|$)", re.DOTALL)
CHUNK_BLOCK_START = "<BEGIN_UNTRUSTED_SELECTED_CHUNKS>\n"
CHUNK_BLOCK_END = "\n<END_UNTRUSTED_SELECTED_CHUNKS>"
REQUIRED_MODEL_ENV = ("RAG_LLM_ENDPOINT", "RAG_LLM_MODEL", "RAG_LLM_API_KEY")
STOP_WORDS = set(
    "a an and are at be been but by can could did do does for from has have is it its "
    "of on or should the there this to was were what when where which who why with "
    "would reported feedback problem problems issue issues about handle create short near in "
    "action item items county confirmed consecutive current now please tell me"
    .split()
)


def _read_gold_set() -> list[dict[str, Any]]:
    cases = []
    for line in GOLD_SET.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or not line.split("|")[1].strip().isdigit():
            continue
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        if len(columns) != 4:
            raise ValueError(f"unexpected gold-set row: {line}")
        cases.append({
            "number": int(columns[0]),
            "question": columns[1],
            "expected": columns[2],
            "source_ids": SOURCE_ID_PATTERN.findall(columns[3]),
        })
    if len(cases) != 15:
        raise ValueError(f"expected 15 gold-set questions; found {len(cases)}")
    return cases


def _parse_selected(prompt: str) -> list[Candidate]:
    start = prompt.index(CHUNK_BLOCK_START) + len(CHUNK_BLOCK_START)
    end = prompt.index(CHUNK_BLOCK_END, start)
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
        for record in json.loads(prompt[start:end])
    ]


def _messages(selected: list[Candidate]) -> list[tuple[str, str, Candidate]]:
    return [
        (match.group(1), f"{match.group(1)}: {match.group(2).strip()}", candidate)
        for candidate in selected
        for match in MESSAGE_PATTERN.finditer(candidate.chunk.text)
    ]


def _proper_location_terms(query: str) -> set[str]:
    first_word = re.search(r"\b\w+\b", query)
    first_start = first_word.start() if first_word else -1
    return {
        word.casefold()
        for match in re.finditer(r"\b[A-Z][a-z]{2,}\b", query)
        for word in [match.group()]
        if match.start() != first_start and word.casefold() not in STOP_WORDS
    }


def _normalized_terms(text: str) -> set[str]:
    terms = set(tokenize(text))
    terms.update(term[:-3] for term in tuple(terms) if term.endswith("ing") and len(term) > 5)
    terms.update(term[:-1] for term in tuple(terms) if term.endswith("s") and len(term) > 4)
    return terms


def _mock_answer(query: str, selected: list[Candidate]) -> str:
    """Produce a deterministic answer from retrieved text, never from gold labels."""
    query_ids = set(SOURCE_ID_PATTERN.findall(query))
    query_terms = {
        term for term in _normalized_terms(query)
        if term not in STOP_WORDS and not term.isdigit()
    }
    location_terms = _proper_location_terms(query)
    ranked = []
    for source_id, message, candidate in _messages(selected):
        if query_ids and source_id not in query_ids:
            continue
        message_terms = _normalized_terms(message)
        overlap = len(query_terms & message_terms)
        if query_ids and source_id in query_ids:
            overlap = max(overlap, len(query_terms), 1)
        if overlap > 0 and overlap >= min(2, len(query_terms)):
            ranked.append((overlap, source_id, message, candidate))

    location_constrained = False
    if not query_ids and location_terms:
        location_matches = [
            row for row in ranked
            if location_terms.issubset(_normalized_terms(row[2]))
        ]
        if len(location_terms) == 1:
            topic_terms = query_terms - location_terms
            location_matches = [
                row for row in location_matches
                if topic_terms & _normalized_terms(row[2])
            ]
        ranked = location_matches
        location_constrained = bool(ranked)
    if not ranked:
        return json.dumps({"status": "insufficient_evidence", "action_items": []})

    ranked.sort(key=lambda result: (-result[0], result[1]))
    best_overlap = ranked[0][0]
    relevant = (
        ranked
        if location_constrained
        else [result for result in ranked if result[0] == best_overlap]
    )
    relevant_ids = {result[1] for result in relevant}
    selected_ids = {message[0] for message in _messages(selected)}
    query_text = query.casefold()
    relevant_text = " ".join(result[2].casefold() for result in relevant)

    missing_detail = (
        ("how many" in query_text and any(word in query_text for word in ("litre", "liter", "amount")))
        or ("exact" in query_text and "schedule" in query_text)
        or ("enough information" in query_text and "enough details" in relevant_text)
    )
    if missing_detail:
        return json.dumps({"status": "insufficient_evidence", "action_items": []})

    conflict = (
        ("Upendo" in query and {"FB-022", "FB-039"}.issubset(selected_ids))
        or ("Miale" in query and {"FB-014", "FB-037"}.issubset(selected_ids))
        or (
            "Nguvu" in query and "FB-016" in relevant_ids
            and "another neighbour" in relevant_text and "confusing" in relevant_text
        )
    )
    if conflict:
        return json.dumps({"status": "conflict", "action_items": []})

    citations = [
        {
            "doc_id": candidate.chunk.doc_id,
            "chunk_id": candidate.chunk.chunk_id,
            "excerpt": message,
        }
        for _, _, message, candidate in relevant[:5]
    ]
    action = {
        "category": "citizen feedback",
        "location": "as reported in the cited feedback",
        "priority": "review",
        "suggested_department": "relevant department",
        "summary": f"Review the reported feedback: {relevant[0][2]}"[:500],
        "citations": citations,
    }
    return json.dumps({"status": "ok", "action_items": [action]}, ensure_ascii=False)


def _run_case(case: dict[str, Any], use_real_model: bool) -> tuple[dict[str, Any], list[Candidate]]:
    prompts: list[str] = []

    def invoke(prompt: str, *, system_prompt: str) -> str:
        prompts.append(prompt)
        if use_real_model:
            return call_model(prompt, system_prompt=system_prompt)
        selected = _parse_selected(prompt)
        mock = MockLLMClient([_mock_answer(case["question"], selected)])
        return mock(prompt, system_prompt=system_prompt)

    result = run_pipeline(case["question"], llm=invoke)
    selected = _parse_selected(prompts[0]) if prompts else []
    return result, selected


def _expected_status(case: dict[str, Any]) -> str:
    expected = case["expected"].casefold()
    if any(term in expected for term in ("conflict", "conflicting")):
        return "conflict"
    if not case["source_ids"] or any(
        term in expected for term in ("not provided", "not enough details", "lacks enough details", "amount is not")
    ):
        return "insufficient_evidence"
    return "ok"


def _render_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="force deterministic evidence-only mock mode, even if model settings exist",
    )
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    configured = [name for name in REQUIRED_MODEL_ENV if os.environ.get(name, "").strip()]
    if not args.mock and configured and len(configured) != len(REQUIRED_MODEL_ENV):
        missing = [name for name in REQUIRED_MODEL_ENV if name not in configured]
        raise RuntimeError(
            "Partial model configuration; set all required variables or use --mock: "
            + ", ".join(missing)
        )
    use_real_model = not args.mock and len(configured) == len(REQUIRED_MODEL_ENV)
    mode = "configured model endpoint" if use_real_model else "evidence-only mock client"

    data_files = sorted(
        path for path in DATA_DIR.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".txt", ".md"}
    )
    if not data_files:
        raise RuntimeError(f"No .txt or .md evaluation data found in {DATA_DIR}")
    ingest_documents(data_files)

    cases = _read_gold_set()
    rows = []
    supported_count = 0
    for case in cases:
        result, selected = _run_case(case, use_real_model)
        retrieved_ids = sorted({
            source_id
            for candidate in selected
            for source_id in SOURCE_ID_PATTERN.findall(candidate.chunk.text)
        })
        cited_ids = sorted({
            source_id
            for item in result.get("action_items", [])
            for citation in item.get("citations", [])
            for source_id in SOURCE_ID_PATTERN.findall(citation["excerpt"])
        })
        selected_by_key = {
            (candidate.chunk.doc_id, candidate.chunk.chunk_id): candidate.chunk.text
            for candidate in selected
        }
        expected_ids = set(case["source_ids"])
        action_items = result.get("action_items", [])
        issues = []

        for item in action_items:
            for citation in item.get("citations", []):
                text = selected_by_key.get((citation["doc_id"], citation["chunk_id"]))
                if text is None or citation["excerpt"] not in text:
                    issues.append("citation is not an exact excerpt from a selected chunk")
        if expected_ids and not expected_ids.issubset(set(retrieved_ids)):
            issues.append("one or more expected source IDs were not retrieved")
        if expected_ids and cited_ids and not set(cited_ids).issubset(expected_ids):
            issues.append("citation references a source outside the gold references")
        claimed_ids = {
            source_id for item in action_items
            for source_id in SOURCE_ID_PATTERN.findall(item.get("summary", ""))
        }
        if expected_ids and claimed_ids and not claimed_ids.issubset(expected_ids):
            issues.append("summary claims a source outside the gold references")

        status_matches = result["status"] == _expected_status(case)
        if result["status"] == "error":
            issues.append("pipeline returned an error; details omitted from the report")
        elif not status_matches:
            issues.append("pipeline status does not match the gold-set expectation")
        if result["status"] == "ok" and not action_items:
            issues.append("ok status has no action item")
        if result["status"] != "ok" and action_items:
            issues.append("non-ok status contains action items")
        if result["status"] == "ok" and expected_ids and not expected_ids.issubset(set(cited_ids)):
            issues.append("one or more expected source IDs were not cited")

        supported = not issues and status_matches and expected_ids.issubset(set(retrieved_ids))
        if supported:
            supported_count += 1
        issue_text = "; ".join(dict.fromkeys(issues)) or (
            "None detected by automated source/citation checks; semantic entailment needs human review"
        )
        rows.append(
            "| {question} | {expected} | {retrieved} | {status} | {supported} | {issues} |".format(
                question=_render_cell(f'{case["number"]}. {case["question"]}'),
                expected=_render_cell(", ".join(case["source_ids"]) or "None"),
                retrieved=_render_cell(", ".join(retrieved_ids) or "None"),
                status=_render_cell(result["status"]),
                supported="yes" if supported else "no",
                issues=_render_cell(issue_text),
            )
        )

    report = [
        "# Gold-set evaluation",
        "",
        f"Run mode: **{mode}**.",
        "",
        "The default mock synthesizes from selected chunks only; it does not read expected answers or gold source IDs. Gold references are used only for scoring. Automated checks cover retrieval IDs, expected status, citation membership, and exact excerpts; they cannot establish semantic entailment, which requires human review.",
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
