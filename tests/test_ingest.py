from pathlib import Path

import pytest

from ingest import IngestionError, ingest_directory


def test_noisy_file_is_cleaned_and_original_is_untouched(tmp_path):
    source = tmp_path / "feedback.md"
    original = (
        "# Roads\n\n"
        "Pothole at <b>Main Street</b>!!!! \U0001f600\n\n"
        "   Water    leak reported.   \n"
    )
    source.write_text(original, encoding="utf-8")

    chunks = ingest_directory(tmp_path)

    assert len(chunks) == 2
    assert chunks[0].section == "Roads"
    assert chunks[0].text == "Pothole at Main Street!!"
    assert chunks[1].text == "Water leak reported."
    assert source.read_text(encoding="utf-8") == original


def test_default_data_directory_is_resolved_relative_to_ingest_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    chunks = ingest_directory()

    assert chunks
    assert any("Mwangaza Road" in chunk.text for chunk in chunks)


def test_empty_file_reports_explicit_ingestion_failure(tmp_path):
    source = tmp_path / "empty.txt"
    source.write_text(" \n\t ", encoding="utf-8")

    with pytest.raises(IngestionError, match="file is empty") as error:
        ingest_directory(tmp_path)

    assert error.value.path == source


def test_unreadable_file_reports_explicit_ingestion_failure(tmp_path, monkeypatch):
    source = tmp_path / "unreadable.txt"
    source.write_text("feedback text", encoding="utf-8")
    original_read_text = Path.read_text

    def fail_read(path, *args, **kwargs):
        if path == source:
            raise PermissionError("simulated access denial")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read)

    with pytest.raises(IngestionError, match="unreadable file") as error:
        ingest_directory(tmp_path)

    assert error.value.path == source
    assert "simulated access denial" in error.value.reason


def test_chunk_ids_are_stable_across_ingestion_runs(tmp_path):
    source = tmp_path / "citizen-feedback.txt"
    source.write_text(
        "Road damage reported at Kijiji Road.\n\n"
        "Water outage reported near the market.\n",
        encoding="utf-8",
    )

    first = ingest_directory(tmp_path)
    second = ingest_directory(tmp_path)

    assert [(chunk.doc_id, chunk.chunk_id) for chunk in first] == [
        (chunk.doc_id, chunk.chunk_id) for chunk in second
    ]
