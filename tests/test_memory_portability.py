import io
import json
import re
import tarfile
import types
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Barrier

import pytest

from memory_portability.archive import unpack
from memory_portability.errors import (
    ArchiveValidationError,
    MemoryImportError,
)
from memory_portability import MemoryStore, MemoryTransfer, exporter
from memory_portability.records import profiles_compatible
from memory_portability.validator import load_and_validate, sha256_file


PROFILE = {"provider": "Local", "model": "test-model", "vector_dimension": 2}


def make_store(tmp_path, name, embed_batch=None, store_class=MemoryStore):
    return store_class(
        memory_dir=tmp_path / name / "memory",
        chroma_path=tmp_path / name / "chroma",
        embed_batch=embed_batch,
        embedding_profile=PROFILE,
    )


def record(record_id, document, embedding=None, metadata=None):
    return {
        "id": record_id,
        "document": document,
        "embedding": embedding or [0.25, 0.75],
        "metadata": metadata or {"time": "2026-08-20 10:00:00"},
    }


def user_records(store):
    return [item for batch in store.iter_user_records() for item in batch]


def export_concurrently(monkeypatch, transfer, filename=None):
    barrier = Barrier(2, timeout=10)
    read_history = transfer.store.read_history

    def synchronized_read():
        barrier.wait()
        return read_history()

    monkeypatch.setattr(transfer.store, "read_history", synchronized_read)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(transfer.export, "history", filename=filename)
            for _ in range(2)
        ]
    return futures


def test_round_trip_restores_user_memory_and_preserves_knowledge(tmp_path):
    source = make_store(tmp_path, "source")
    source.write_history("source history\n")
    source.upsert_records([record("source-memory", "remember me")])

    transfer = MemoryTransfer(tmp_path / "transfer", source)
    exported = transfer.export("both", filename="memory.tar.gz")

    with tarfile.open(tmp_path / "transfer" / exported["filename"], "r:gz") as archive:
        assert set(archive.getnames()) == {
            "manifest.json",
            "history/history.metta",
            "vector/collections.json",
            "vector/records.jsonl",
        }

    target = make_store(tmp_path, "target")
    target.write_history("old history\n")
    target.upsert_records(
        [
            record("old-memory", "replace me"),
            record(
                "knowledge-prior",
                "keep me",
                metadata={"type": "chunk", "source": "knowledge.md"},
            ),
        ]
    )

    result = MemoryTransfer(tmp_path / "transfer", target).import_archive(
        "memory.tar.gz", mode="overwrite"
    )

    assert result["status"] == "imported"
    assert target.read_history() == "source history\n"
    assert [item["id"] for item in user_records(target)] == ["source-memory"]
    assert target.collection().get(ids=["knowledge-prior"])["ids"] == ["knowledge-prior"]


@pytest.mark.parametrize("concurrent", [False, True], ids=["sequential", "overlapping"])
def test_same_second_exports_have_unique_filenames(tmp_path, monkeypatch, concurrent):
    source = make_store(tmp_path, "source")
    source.write_history("history\n")
    transfer_dir = tmp_path / "transfer"
    transfer = MemoryTransfer(transfer_dir, source)
    fixed_time = datetime(2026, 10, 12, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        exporter, "datetime", types.SimpleNamespace(now=lambda tz: fixed_time)
    )

    if concurrent:
        results = [future.result() for future in export_concurrently(monkeypatch, transfer)]
    else:
        results = [transfer.export("history") for _ in range(2)]

    assert len({result["filename"] for result in results}) == 2
    for index, result in enumerate(results):
        assert re.fullmatch(
            r"omega-memory-20261012T120000Z-[0-9a-f]{8}\.tar\.gz", result["filename"]
        )
        assert sha256_file(transfer_dir / result["filename"]) == result["sha256"]
        target = make_store(tmp_path, f"target-{index}")
        MemoryTransfer(transfer_dir, target).import_archive(result["filename"])
        assert target.read_history() == "history\n"
    assert set(transfer_dir.iterdir()) == {
        transfer_dir / result["filename"] for result in results
    }


def test_concurrent_exports_do_not_overwrite_same_filename(tmp_path, monkeypatch):
    source = make_store(tmp_path, "source")
    source.write_history("history\n")
    transfer_dir = tmp_path / "transfer"
    transfer = MemoryTransfer(transfer_dir, source)

    futures = export_concurrently(monkeypatch, transfer, filename="same.tar.gz")

    assert sum(isinstance(future.exception(), FileExistsError) for future in futures) == 1
    results = [future.result() for future in futures if future.exception() is None]
    assert len(results) == 1
    assert results[0]["filename"] == "same.tar.gz"
    archive_path = transfer_dir / "same.tar.gz"
    assert sha256_file(archive_path) == results[0]["sha256"]
    assert list(transfer_dir.iterdir()) == [archive_path]


def test_failed_export_publication_cleans_temporary_files(tmp_path, monkeypatch):
    source = make_store(tmp_path, "source")
    source.write_history("history\n")
    transfer_dir = tmp_path / "transfer"

    def fail_link(source_path, destination):
        raise OSError("hard links unavailable")

    monkeypatch.setattr(exporter.os, "link", fail_link)

    with pytest.raises(OSError, match="hard links unavailable"):
        MemoryTransfer(transfer_dir, source).export("history")

    assert list(transfer_dir.iterdir()) == []


def test_export_without_embeddings_reembeds_on_import(tmp_path):
    source = make_store(tmp_path, "source")
    source.upsert_records([record("source-memory", "portable text")])
    transfer_dir = tmp_path / "transfer"
    MemoryTransfer(transfer_dir, source).export(
        "ltm", include_embeddings=False, filename="without-embeddings.tar.gz"
    )

    calls = []

    def embed_batch(documents):
        calls.append(documents)
        return [[0.5, 0.5] for _ in documents]

    target = make_store(tmp_path, "target", embed_batch=embed_batch)
    result = MemoryTransfer(transfer_dir, target).import_archive(
        "without-embeddings.tar.gz", include_history=False
    )

    assert result["reembedded"] is True
    assert calls == [["portable text"]]
    assert user_records(target)[0]["embedding"] == pytest.approx([0.5, 0.5])


def test_append_namespaces_ids_and_preserves_existing_records(tmp_path):
    source = make_store(tmp_path, "source")
    source.write_history("imported history\n")
    source.upsert_records([record("shared-id", "imported")])
    transfer_dir = tmp_path / "transfer"
    MemoryTransfer(transfer_dir, source).export("both", filename="append.tar.gz")

    target = make_store(tmp_path, "target")
    target.write_history("existing history\n")
    target.upsert_records([record("shared-id", "existing")])
    MemoryTransfer(transfer_dir, target).import_archive("append.tar.gz", mode="append")

    records = user_records(target)
    assert len(records) == 2
    assert {item["document"] for item in records} == {"existing", "imported"}
    assert any(item["id"].startswith("import-") for item in records)
    assert target.read_history() == "existing history\nimported history\n"


def test_unpack_rejects_unsafe_member(tmp_path):
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("../history.metta")
        payload = b"private"
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(ArchiveValidationError, match="Unexpected archive member"):
        unpack(archive_path, tmp_path / "unpacked")


class FailOnceStore(MemoryStore):
    fail_next_smoke_test = True

    def smoke_test(self, history, vectors):
        if self.fail_next_smoke_test:
            self.fail_next_smoke_test = False
            raise RuntimeError("simulated post-import failure")
        super().smoke_test(history, vectors)


class CrashOnceStore(MemoryStore):
    crash_next_smoke_test = True

    def smoke_test(self, history, vectors):
        if self.crash_next_smoke_test:
            self.crash_next_smoke_test = False
            raise KeyboardInterrupt("simulated process interruption")
        super().smoke_test(history, vectors)


def test_failed_overwrite_restores_previous_memory(tmp_path):
    source = make_store(tmp_path, "source")
    source.write_history("source history\n")
    source.upsert_records([record("source-memory", "source")])
    transfer_dir = tmp_path / "transfer"
    MemoryTransfer(transfer_dir, source).export("both", filename="rollback.tar.gz")

    target = make_store(tmp_path, "target", store_class=FailOnceStore)
    target.write_history("target history\n")
    target.upsert_records([record("target-memory", "target")])

    with pytest.raises(MemoryImportError, match="rolled back"):
        MemoryTransfer(transfer_dir, target).import_archive("rollback.tar.gz")

    assert target.read_history() == "target history\n"
    assert [item["id"] for item in user_records(target)] == ["target-memory"]


def test_failed_overwrite_removes_vector_store_created_for_fresh_memory(tmp_path):
    source = make_store(tmp_path, "source")
    source.write_history("source history\n")
    source.upsert_records([record("source-memory", "source")])
    transfer_dir = tmp_path / "transfer"
    MemoryTransfer(transfer_dir, source).export("both", filename="fresh.tar.gz")

    target = make_store(tmp_path, "fresh-target", store_class=FailOnceStore)
    assert target.history_path.exists() is False
    assert target.chroma_path.exists() is False

    with pytest.raises(MemoryImportError, match="rolled back"):
        MemoryTransfer(transfer_dir, target).import_archive("fresh.tar.gz")

    assert target.history_path.exists() is False
    assert target.chroma_path.exists() is False


def test_unknown_source_dimension_is_not_compatible():
    source = {"provider": "OpenAI", "model": "model", "vector_dimension": None}
    active = {"provider": "OpenAI", "model": "model", "vector_dimension": 3072}

    assert profiles_compatible(source, active, missing_embeddings=False) is False


@pytest.mark.parametrize("missing_field", ["created_at", "source", "embeddings_included"])
def test_manifest_requires_portability_metadata(tmp_path, missing_field):
    source = make_store(tmp_path, f"source-{missing_field}")
    source.write_history("history\n")
    transfer_dir = tmp_path / f"transfer-{missing_field}"
    result = MemoryTransfer(transfer_dir, source).export(
        "history", filename=f"{missing_field}.tar.gz"
    )

    staging = tmp_path / f"staging-{missing_field}"
    members = unpack(transfer_dir / result["filename"], staging)
    manifest_path = staging / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop(missing_field)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArchiveValidationError, match=missing_field):
        load_and_validate(staging, members)


def test_recover_rolls_back_an_interrupted_overwrite(tmp_path):
    source = make_store(tmp_path, "source")
    source.write_history("source history\n")
    source.upsert_records([record("source-memory", "source")])
    transfer_dir = tmp_path / "transfer"
    MemoryTransfer(transfer_dir, source).export("both", filename="recovery.tar.gz")

    target = make_store(tmp_path, "target", store_class=CrashOnceStore)
    target.write_history("target history\n")
    target.upsert_records([record("target-memory", "target")])
    transfer = MemoryTransfer(transfer_dir, target)

    with pytest.raises(KeyboardInterrupt):
        transfer.import_archive("recovery.tar.gz")

    assert transfer.recover() == {"status": "rolled-back"}
    assert target.read_history() == "target history\n"
    assert [item["id"] for item in user_records(target)] == ["target-memory"]
