"""Tests for SAYT storage helper validation and artifact edge cases."""

# pylint: disable=protected-access,too-few-public-methods,missing-function-docstring,no-member

import json
from contextlib import contextmanager

import pytest

from survey_assist_embed_core.sayt import (
    PrefixRetrieverSpec,
    SemanticRetrieverSpec,
    retriever_specs,
    storage,
)
from survey_assist_embed_core.sayt.core import CleanCorpus
from survey_assist_embed_core.sayt.weight_specs import (
    NgramWeightSpec,
    PrefixWeightSpec,
    SemanticWeightSpec,
    WeightSpecs,
)


def test_prepare_artifact_dir_handles_existing_paths(tmp_path):
    """Reject accidental reuse, then replace existing directories or files."""
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    stale_file = artifact_dir / "stale.txt"
    stale_file.write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Artifact directory already exists"):
        storage.prepare_artifact_dir(artifact_dir)

    result = storage.prepare_artifact_dir(artifact_dir, overwrite=True)

    assert result == artifact_dir
    assert artifact_dir.is_dir()
    assert not stale_file.exists()

    artifact_file = tmp_path / "artifact-file"
    artifact_file.write_text("stale", encoding="utf-8")

    replaced = storage.prepare_artifact_dir(artifact_file, overwrite=True)

    assert replaced == artifact_file
    assert artifact_file.is_dir()


def test_load_corpus_from_csv_downloads_gcs_inputs(monkeypatch, tmp_path):
    """Resolve GCS CSV inputs to a local file before reading them."""
    csv_path = tmp_path / "source.csv"
    csv_path.write_text("title,display\ndog,Dog\n", encoding="utf-8")

    @contextmanager
    def fake_resolve_local_path(_path: str):
        yield str(csv_path)

    monkeypatch.setattr(storage, "resolve_local_path", fake_resolve_local_path)

    rows = storage.load_corpus_from_csv(
        "gs://bucket/source.csv",
        display_text_col="display",
    )

    assert rows == [("dog", "Dog")]


def test_read_artifact_inputs_validate_missing_and_malformed_state(tmp_path):
    """Raise clear errors for missing files and malformed manifest payloads."""
    with pytest.raises(FileNotFoundError, match="Artifact corpus file not found"):
        storage.read_artifact_corpus(artifact_dir=tmp_path)

    with pytest.raises(FileNotFoundError, match="Artifact manifest not found"):
        storage.read_artifact_manifest(artifact_dir=tmp_path)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"artifact_type": "other", "artifact_version": 2}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported artifact type"):
        storage.read_artifact_manifest(artifact_dir=tmp_path)

    manifest_path.write_text(
        json.dumps({"artifact_type": "sayt", "artifact_version": 999}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported artifact version"):
        storage.read_artifact_manifest(artifact_dir=tmp_path)

    manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "sayt",
                "artifact_version": 2,
                "min_chars": 3,
                "corpus_file": "corpus.csv",
                "corpus_size": 1,
                "retrievers": [],
                "weights": [{"type": "prefix", "weights": 1}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="Malformed artifact manifest: missing max_suggestions"
    ):
        storage.read_artifact_manifest(artifact_dir=tmp_path)


def test_storage_helper_validation_errors():
    """Guard helper APIs against invalid types, paths, and unsupported specs."""
    stored_retriever = storage.StoredRetrieverSpec(
        spec=PrefixRetrieverSpec(),
        path=None,
    )

    with pytest.raises(
        ValueError, match="Retriever 'ngram' does not have a stored filespace"
    ):
        retriever_specs._require_filespace_path(None, spec_name="ngram")

    with pytest.raises(ValueError, match="does not have a stored filespace"):
        storage.retriever_filespace_path("artifact", stored_retriever)

    with pytest.raises(ValueError, match="Malformed retriever config for type: prefix"):
        storage._deserialise_stored_retriever(
            {"type": "prefix", "weight": 1.0, "config": []}
        )

    with pytest.raises(ValueError, match="Unsupported stored retriever type: missing"):
        storage._deserialise_stored_retriever(
            {"type": "missing", "weight": 1.0, "config": {}}
        )

    class _UnknownSpec:
        name = "unknown"

        def build(self, corpus, *, min_chars):
            _ = (corpus, min_chars)

    with pytest.raises(
        TypeError,
        match="Only artifact-aware retriever specs can be persisted; got _UnknownSpec",
    ):
        storage._build_stored_retriever(0, _UnknownSpec())

    with pytest.raises(
        ValueError, match="Malformed integer value for retriever field: n"
    ):
        storage._coerce_int(True, field_name="n")


def test_serialise_stored_retriever_uses_object_dict_for_non_dataclass_specs():
    """Serialise runtime-only spec config from a plain object's __dict__."""

    class _CustomSpec:
        def __init__(self):
            self.name = "custom"
            self.trigger = "groom"
            self.limit = 3

        def build(self, corpus, *, min_chars):
            _ = (corpus, min_chars)

        def load_from_artifact(self, corpus, *, min_chars, filespace_path):
            _ = (corpus, min_chars, filespace_path)

    stored_retriever = storage.StoredRetrieverSpec(
        spec=_CustomSpec(),
        path="retrievers/99-custom",
    )

    assert storage._serialise_stored_retriever(stored_retriever) == {
        "type": "custom",
        "path": "retrievers/99-custom",
        "config": {
            "trigger": "groom",
            "limit": 3,
        },
    }


def test_weight_specs_round_trip_through_storage():
    """Preserve weight names, values, and order through serialization."""
    original = WeightSpecs(
        specs=[
            PrefixWeightSpec(weights=2.0),
            NgramWeightSpec(weights={1: 0.5, 4: 1.5}),
        ]
    )

    payload = storage._serialise_weight_specs(original)
    restored = storage._deserialise_weight_specs(payload)

    assert restored.specs == original.specs


def test_empty_weight_specs_round_trip_through_storage():
    """Preserve an empty weight-spec collection through serialization."""
    original = WeightSpecs(specs=[])

    payload = storage._serialise_weight_specs(original)
    restored = storage._deserialise_weight_specs(payload)

    assert not payload
    assert not restored.specs


def test_all_weight_spec_types_round_trip_through_storage():
    """Restore each supported weight-spec subtype from serialized data."""
    original = WeightSpecs(
        specs=[
            PrefixWeightSpec(weights=1.0),
            NgramWeightSpec(weights=2.0),
            SemanticWeightSpec(weights=3.0),
        ]
    )

    restored = storage._deserialise_weight_specs(
        storage._serialise_weight_specs(original)
    )

    assert [type(spec) for spec in restored.specs] == [
        PrefixWeightSpec,
        NgramWeightSpec,
        SemanticWeightSpec,
    ]
    assert restored.specs == original.specs


def test_deserialise_weight_spec_defaults_missing_weights():
    """Use the default weight when the serialized field is absent."""
    restored = storage._deserialise_weight_specs([{"retriever_name": "prefix"}])

    assert restored.specs == [
        PrefixWeightSpec(),
    ]


@pytest.mark.parametrize(
    "payload, message",
    [
        ({}, "Weight specs payload must be a list"),
        ([None], "Each weight spec item must be a dictionary"),
        ([{"retriever_name": "unknown"}], "Unsupported weight spec type: unknown"),
    ],
)
def test_deserialise_weight_specs_rejects_malformed_payload(payload, message):
    """Reject malformed or unsupported serialized weight specifications."""
    with pytest.raises(ValueError, match=message):
        storage._deserialise_weight_specs(payload)


def test_semantic_retriever_artifact_round_trips_and_loads(
    monkeypatch, tmp_path, small_corpus
):
    """Round-trip semantic artifact state and delegate dense index load/build calls."""
    captured = {}
    corpus = CleanCorpus.model_validate(small_corpus)
    spec = SemanticRetrieverSpec(
        model="all-MiniLM-L6-v2",
        vectoriser_class="OnnxVectoriser",
    )
    stored_retriever = storage._build_stored_retriever(2, spec)
    path = tmp_path / stored_retriever.path

    def _fake_build_semantic_index(
        corpus_arg, *, model, vectoriser_class, output_dir, overwrite
    ):
        captured["build"] = {
            "corpus": corpus_arg,
            "model": model,
            "vectoriser_class": vectoriser_class,
            "output_dir": output_dir,
            "overwrite": overwrite,
        }

    def _fake_load_semantic_index(corpus_arg, *, model, vectoriser_class, folder_path):
        captured["load"] = {
            "corpus": corpus_arg,
            "model": model,
            "vectoriser_class": vectoriser_class,
            "folder_path": folder_path,
        }
        return "loaded-index"

    class _StubSemanticRetriever:
        @classmethod
        def from_index(cls, corpus_arg, *, min_chars, index):
            captured["from_index"] = {
                "corpus": corpus_arg,
                "min_chars": min_chars,
                "index": index,
            }
            return {"index": index, "min_chars": min_chars}

    monkeypatch.setattr(
        retriever_specs, "build_semantic_index", _fake_build_semantic_index
    )
    monkeypatch.setattr(
        retriever_specs, "load_semantic_index", _fake_load_semantic_index
    )
    monkeypatch.setattr(retriever_specs, "SemanticRetriever", _StubSemanticRetriever)

    rebuilt = storage._deserialise_stored_retriever(
        {
            "type": stored_retriever.spec.name,
            "path": stored_retriever.path,
            "config": {
                "model": "all-MiniLM-L6-v2",
                "vectoriser_class": "OnnxVectoriser",
            },
        }
    )

    assert stored_retriever.spec.name == "semantic"
    assert stored_retriever.path == "retrievers/02-semantic"
    assert isinstance(rebuilt.spec, SemanticRetrieverSpec)
    assert rebuilt.spec.vectoriser_class == "OnnxVectoriser"

    storage.build_retriever_artifact(
        corpus=corpus,
        min_chars=3,
        stored_retriever=stored_retriever,
        artifact_dir=tmp_path,
    )
    retriever = storage.load_retriever_from_artifact(
        corpus=corpus,
        min_chars=3,
        stored_retriever=stored_retriever,
        artifact_dir=tmp_path,
    )

    assert retriever == {"index": "loaded-index", "min_chars": 3}
    assert captured == {
        "build": {
            "corpus": corpus,
            "model": "all-MiniLM-L6-v2",
            "vectoriser_class": "OnnxVectoriser",
            "output_dir": path,
            "overwrite": True,
        },
        "load": {
            "corpus": corpus,
            "model": "all-MiniLM-L6-v2",
            "vectoriser_class": "OnnxVectoriser",
            "folder_path": path,
        },
        "from_index": {
            "corpus": corpus,
            "min_chars": 3,
            "index": "loaded-index",
        },
    }
