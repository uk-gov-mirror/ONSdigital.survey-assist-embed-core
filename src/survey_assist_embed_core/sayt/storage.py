"""Artifact and storage helpers for SAYT builder and loader paths."""

import csv
import json
import os
import shutil
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

import pandas as pd

from survey_assist_embed_core.adapters.storage import resolve_local_path
from survey_assist_embed_core.sayt.core import (
    CleanCorpus,
    PersistedCorpusRow,
    validate_max_suggestions,
    validate_min_chars,
)
from survey_assist_embed_core.sayt.retriever_specs import (
    ArtifactRetrieverSpec,
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    Retriever,
    RetrieverSpec,
    SemanticRetrieverSpec,
)
from survey_assist_embed_core.sayt.weight_specs import (
    NgramWeightSpec,
    PrefixWeightSpec,
    SemanticWeightSpec,
    WeightConfig,
    WeightSpecs,
)

SAYT_ARTIFACT_TYPE = "sayt"
SAYT_ARTIFACT_VERSION = 2
MANIFEST_FILE_NAME = "manifest.json"
CORPUS_FILE_NAME = "corpus.csv"
_ARTIFACT_CORPUS_FIELDS = ["search_text", "display_text"]


@dataclass(frozen=True, slots=True)
class StoredRetrieverSpec:
    """Persisted retriever spec plus its optional filespace path."""

    spec: ArtifactRetrieverSpec
    path: str | None = None


@dataclass(frozen=True, slots=True)
class SaytArtifactManifest:
    """Structured manifest data for a persisted SAYT artifact."""

    min_chars: int
    max_suggestions: int
    corpus_file: str
    corpus_size: int
    retrievers: tuple[StoredRetrieverSpec, ...]
    weight_specs: WeightSpecs


def load_corpus_from_csv(
    file_path: str | os.PathLike[str],
    *,
    search_text_col: str = "title",
    display_text_col: str | None = None,
) -> list[tuple[object, object]]:
    """Load raw corpus tuples from a CSV file.

    Args:
        file_path: Local path or ``gs://`` URI for the CSV file containing
            suggestion rows.
        search_text_col: Column containing the searchable text.
        display_text_col: Optional column containing display text. When
            omitted, the search column is reused for display values.

    Returns:
        Raw ``(search_text, display_text)`` tuples suitable for ``CleanCorpus``.

    Raises:
        ValueError: If the requested search or display column is missing.
    """
    with resolve_local_path(str(file_path)) as local_file:
        df = pd.read_csv(local_file)
    if search_text_col not in df.columns:
        raise ValueError(f"Column '{search_text_col}' not found in CSV")
    if display_text_col is None:
        display_text_col = search_text_col
    if display_text_col not in df.columns:
        raise ValueError(f"Column '{display_text_col}' not found in CSV")
    return list(zip(df[search_text_col], df[display_text_col], strict=False))


def prepare_artifact_dir(
    artifact_dir: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> Path:
    """Create or replace the output directory for a SAYT artifact."""
    path = Path(artifact_dir)
    if path.exists():
        if not overwrite:
            raise FileExistsError("Artifact directory already exists")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_artifact_corpus(corpus: CleanCorpus, *, artifact_dir: str | Path) -> Path:
    """Persist cleaned SAYT rows as the artifact corpus source of truth."""
    output_path = Path(artifact_dir) / CORPUS_FILE_NAME
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_ARTIFACT_CORPUS_FIELDS)
        writer.writeheader()
        writer.writerows(
            {
                "search_text": search_text,
                "display_text": display_text,
            }
            for search_text, display_text in corpus.rows
        )
    return output_path


def read_artifact_corpus(
    *,
    artifact_dir: str | Path,
    corpus_file: str = CORPUS_FILE_NAME,
) -> list[PersistedCorpusRow]:
    """Read persisted corpus rows from a SAYT artifact."""
    corpus_path = Path(artifact_dir) / corpus_file
    if not corpus_path.exists():
        raise FileNotFoundError(f"Artifact corpus file not found: {corpus_path}")

    with open(corpus_path, encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return [
            PersistedCorpusRow(
                search_text=row["search_text"],
                display_text=row["display_text"],
            )
            for row in reader
        ]


def build_artifact_manifest(
    *,
    corpus: CleanCorpus,
    min_chars: int,
    max_suggestions: int,
    retriever_specs: tuple[RetrieverSpec, ...],
    weight_specs: WeightSpecs,
) -> SaytArtifactManifest:
    """Build the structured manifest payload for a SAYT artifact."""
    return SaytArtifactManifest(
        min_chars=min_chars,
        max_suggestions=max_suggestions,
        corpus_file=CORPUS_FILE_NAME,
        corpus_size=corpus.size,
        retrievers=tuple(
            _build_stored_retriever(index, spec)
            for index, spec in enumerate(retriever_specs)
        ),
        weight_specs=weight_specs,
    )


def write_artifact_manifest(
    manifest: SaytArtifactManifest,
    *,
    artifact_dir: str | Path,
) -> Path:
    """Write the manifest for a SAYT artifact."""
    manifest_path = Path(artifact_dir) / MANIFEST_FILE_NAME
    manifest_path.write_text(
        json.dumps(_serialise_manifest(manifest), indent=2),
        encoding="utf-8",
    )
    return manifest_path


def read_artifact_manifest(*, artifact_dir: str | Path) -> SaytArtifactManifest:
    """Read and validate a SAYT artifact manifest."""
    manifest_path = Path(artifact_dir) / MANIFEST_FILE_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Artifact manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != SAYT_ARTIFACT_TYPE:
        raise ValueError("Unsupported artifact type")
    if payload.get("artifact_version") != SAYT_ARTIFACT_VERSION:
        raise ValueError("Unsupported artifact version")

    try:
        return SaytArtifactManifest(
            min_chars=validate_min_chars(payload["min_chars"]),
            max_suggestions=validate_max_suggestions(payload["max_suggestions"]),
            corpus_file=str(payload["corpus_file"]),
            corpus_size=int(payload["corpus_size"]),
            retrievers=tuple(
                _deserialise_stored_retriever(item) for item in payload["retrievers"]
            ),
            weight_specs=_deserialise_weight_specs(payload["weight_specs"]),
        )
    except KeyError as exc:
        raise ValueError(f"Malformed artifact manifest: missing {exc.args[0]}") from exc


def retriever_filespace_path(
    artifact_dir: str | Path,
    stored_retriever: StoredRetrieverSpec,
) -> Path:
    """Resolve the persisted filespace for a dense retriever entry."""
    if stored_retriever.path is None:
        raise ValueError(
            f"Retriever '{stored_retriever.spec.name}' does not have a stored filespace"
        )
    return Path(artifact_dir) / stored_retriever.path


def _require_artifact_spec(spec: RetrieverSpec) -> ArtifactRetrieverSpec:
    if not isinstance(spec, ArtifactRetrieverSpec):
        raise TypeError(
            "Only artifact-aware retriever specs can be persisted; "
            f"got {type(spec).__name__}"
        )
    return spec


def build_retriever_artifact(
    *,
    corpus: CleanCorpus,
    min_chars: int,
    stored_retriever: StoredRetrieverSpec,
    artifact_dir: str | Path,
) -> None:
    """Persist built-in retriever assets required by a SAYT artifact."""
    if stored_retriever.path is None:
        return

    stored_retriever.spec.build(
        corpus,
        min_chars=min_chars,
        filespace_path=retriever_filespace_path(artifact_dir, stored_retriever),
        overwrite=True,
    )


def load_retriever_from_artifact(
    *,
    corpus: CleanCorpus,
    min_chars: int,
    stored_retriever: StoredRetrieverSpec,
    artifact_dir: str | Path,
) -> Retriever:
    """Restore a built-in runtime retriever from persisted artifact state."""
    return stored_retriever.spec.load_from_artifact(
        corpus,
        min_chars=min_chars,
        filespace_path=(
            retriever_filespace_path(artifact_dir, stored_retriever)
            if stored_retriever.path is not None
            else None
        ),
    )


def _build_stored_retriever(
    index: int,
    spec: RetrieverSpec,
) -> StoredRetrieverSpec:
    artifact_spec = _require_artifact_spec(spec)

    return StoredRetrieverSpec(
        spec=artifact_spec,
        path=(
            None
            if isinstance(artifact_spec, PrefixRetrieverSpec)
            else f"retrievers/{index:02d}-{artifact_spec.name}"
        ),
    )


def _serialise_manifest(manifest: SaytArtifactManifest) -> dict[str, object]:
    return {
        "artifact_type": SAYT_ARTIFACT_TYPE,
        "artifact_version": SAYT_ARTIFACT_VERSION,
        "min_chars": manifest.min_chars,
        "max_suggestions": manifest.max_suggestions,
        "corpus_file": manifest.corpus_file,
        "corpus_size": manifest.corpus_size,
        "retrievers": [
            _serialise_stored_retriever(stored_retriever)
            for stored_retriever in manifest.retrievers
        ],
        "weight_specs": _serialise_weight_specs(manifest.weight_specs),
    }


def _serialise_stored_retriever(
    stored_retriever: StoredRetrieverSpec,
) -> dict[str, object]:
    spec = stored_retriever.spec

    if is_dataclass(spec):
        config: dict[str, object] = {
            field.name: getattr(spec, field.name)
            for field in fields(spec)
            if field.name not in {"name"}
        }
    else:
        raw_config = getattr(spec, "__dict__", None)
        config = (
            {
                str(key): value
                for key, value in raw_config.items()
                if key not in {"name"}
            }
            if isinstance(raw_config, dict)
            else {}
        )

    return {
        "type": spec.name,
        "path": stored_retriever.path,
        "config": config,
    }


def _deserialise_stored_retriever(payload: dict[str, object]) -> StoredRetrieverSpec:
    retriever_type = str(payload["type"])
    path = payload.get("path")
    config = payload.get("config", {})
    if not isinstance(config, dict):
        raise ValueError(f"Malformed retriever config for type: {retriever_type}")
    spec: RetrieverSpec
    if retriever_type == "prefix":
        spec = PrefixRetrieverSpec()
    elif retriever_type == "ngram":
        spec = NgramRetrieverSpec(
            n=_coerce_int(config["n"], field_name="n"),
            max_df=_coerce_float(config["max_df"], field_name="max_df"),
        )
    elif retriever_type == "semantic":
        vectoriser_class = config.get("vectoriser_class")
        spec = SemanticRetrieverSpec(
            model=str(config["model"]),
            vectoriser_class=(
                None if vectoriser_class is None else str(vectoriser_class)
            ),
        )
    else:
        raise ValueError(f"Unsupported stored retriever type: {retriever_type}")

    return StoredRetrieverSpec(
        spec=spec, path=str(path) if isinstance(path, str) else None
    )


def _serialise_weight_specs(weight_specs: WeightSpecs) -> list[dict[str, object]]:
    """Serialize WeightSpecs to a list of dictionaries."""
    serialized: list[dict[str, object]] = []
    for spec in weight_specs.specs:
        serialized.append(
            {
                "retriever_name": spec.retriever_name,
                "weights": spec.weights,
            }
        )
    return serialized


def _deserialise_weight_specs(payload: list[object]) -> WeightSpecs:
    """Deserialize a list of dictionaries back to WeightSpecs."""
    if not isinstance(payload, list):
        raise ValueError("Weight specs payload must be a list")

    specs: list[WeightConfig] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each weight spec item must be a dictionary")

        spec_type = str(item.get("retriever_name", ""))
        weights_value = item.get("weights", 1.0)

        # Deserialize dict weights if present
        if isinstance(weights_value, dict):
            # Convert string keys to int keys
            weights_value = {int(k): float(v) for k, v in weights_value.items()}

        if spec_type == "prefix":
            specs.append(PrefixWeightSpec(weights=weights_value))
        elif spec_type == "ngram":
            specs.append(NgramWeightSpec(weights=weights_value))
        elif spec_type == "semantic":
            specs.append(SemanticWeightSpec(weights=weights_value))
        else:
            raise ValueError(f"Unsupported weight spec type: {spec_type}")

    return WeightSpecs(specs=specs)


def _coerce_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ValueError(f"Malformed integer value for retriever field: {field_name}")
    return int(value)


def _coerce_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"Malformed float value for retriever field: {field_name}")
    return float(value)
