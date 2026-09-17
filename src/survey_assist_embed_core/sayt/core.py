"""Core SAYT data models, corpus cleaning, and ranking helpers."""

# ruff: noqa: PLR2004

import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_SPACE_RE = re.compile(r"[^a-z ]+")


def _normalise(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = _NON_ALNUM_SPACE_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


@dataclass(frozen=True, slots=True)
class PersistedCorpusRow:
    """Represent a persisted SAYT corpus row restored from artifact storage."""

    search_text: str
    display_text: str


class CleanCorpus(BaseModel):
    """Store cleaned and sorted SAYT rows and their derived lookup tables.

    Instances are created from raw strings or ``(search_text, display_text)``
    pairs and expose display-level duplication counts used for ranking.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    rows: list[tuple[str, str]] = Field(default_factory=list)
    size: int = 0
    _display_text_value_counts: dict[str, int] = PrivateAttr(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_input(cls, data: object) -> object:
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            if "rows" in data:
                return data
            if "corpus" in data:
                data = data["corpus"]

        return {
            "rows": cls._clean_corpus(
                cast(Iterable[str] | Iterable[tuple[object, object]], data)
            )
        }

    @property
    def display_text_value_counts(self) -> dict[str, int]:
        """Return per-display-text occurrence counts for the cleaned corpus."""
        return self._display_text_value_counts

    # Pylint does not understand Pydantic's model_post_init signature here.
    def model_post_init(  # pylint: disable=arguments-differ
        self, __context: Any
    ) -> None:
        self._sort_rows()
        self._populate_indexes()

    def _sort_rows(self) -> "CleanCorpus":
        """Sort the cleaned rows by search text and display text."""
        self.rows = sorted(self.rows)
        return self

    def _populate_indexes(self) -> "CleanCorpus":
        """Rebuild lookup tables from the current cleaned rows."""
        self._display_text_value_counts = {}
        for _, display in self.rows:
            self._display_text_value_counts[display] = (
                self._display_text_value_counts.get(display, 0) + 1
            )
        self.size = len(self.rows)
        return self

    @classmethod
    def from_persisted_rows(
        cls,
        rows: Iterable[PersistedCorpusRow | tuple[str, str]],
    ) -> "CleanCorpus":
        """Restore a cleaned corpus from persisted search and display text pairs.

        Args:
            rows: Persisted ``(search_text, display_text)`` pairs or
                ``PersistedCorpusRow`` objects.

        Returns:
            A ``CleanCorpus`` whose rows match the persisted artifact data.

        Raises:
            ValueError: If no persisted rows are supplied.
        """
        restored_rows = [cls._coerce_persisted_row(row) for row in rows]
        if not restored_rows:
            raise ValueError("corpus is empty after filtering")

        return cls.model_construct(rows=restored_rows)

    @staticmethod
    def _coerce_persisted_row(
        row: PersistedCorpusRow | tuple[str, str],
    ) -> tuple[str, str]:
        """Convert persisted row data into the internal tuple format."""
        if isinstance(row, PersistedCorpusRow):
            return (row.search_text, row.display_text)
        search_text, display_text = row
        return (str(search_text), str(display_text))

    @staticmethod
    def _clean_corpus(
        corpus: Iterable[str] | Iterable[tuple[object, object]],
    ) -> list[tuple[str, str]]:
        if not isinstance(corpus, Iterable):
            raise TypeError(
                "corpus must be an iterable of strings or (string, original) tuples"
            )
        cleaned: list[tuple[str, str]] = []
        for item in corpus:
            item_tuple = item if isinstance(item, tuple) else (item, item)
            text = _normalise(item_tuple[0])
            if not text or text == "-9":
                warnings.warn(
                    f"Skipping empty or invalid corpus item: {item!r}",
                    stacklevel=2,
                )
                continue
            display = str(item_tuple[1]).strip()
            if pd.isna(item_tuple[1]) or not display:
                warnings.warn(
                    f"Empty display value for item: {item!r}, using search text as display",
                    stacklevel=2,
                )
                display = str(item_tuple[0]).strip()
            cleaned.append((text, display))
        if not cleaned:
            raise ValueError("corpus is empty after filtering")
        return cleaned


def _coerce_sayt_int_setting(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise TypeError(f"{field_name} must be an integer")
    try:
        return int(value)
    except ValueError as exc:
        raise TypeError(f"{field_name} must be an integer") from exc


def validate_min_chars(value: object) -> int:
    """Validate the global SAYT minimum query length setting."""
    min_chars = _coerce_sayt_int_setting(value, field_name="min_chars")
    if min_chars < 3:
        raise ValueError("min_chars must be >= 3")
    return min_chars


def validate_max_suggestions(value: object) -> int:
    """Validate the global SAYT maximum suggestion count setting."""
    max_suggestions = _coerce_sayt_int_setting(value, field_name="max_suggestions")
    if not 1 <= max_suggestions <= 100:
        raise ValueError("max_suggestions must be between 1 and 100")
    return max_suggestions


class SaytGlobalSettings(BaseModel):
    """Describe suggester-wide runtime settings."""

    model_config = ConfigDict(extra="forbid")

    min_chars: int
    max_suggestions: int


class SaytCorpusSummary(BaseModel):
    """Summarise the cleaned corpus bound to a suggester."""

    model_config = ConfigDict(extra="forbid")

    size: int
    unique_display_texts: int
    max_duplication: int


class SaytRetrieverArtifactProvenance(BaseModel):
    """Capture persisted artifact details for one retriever entry."""

    model_config = ConfigDict(extra="forbid")

    artifact_type: str
    path: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class SaytArtifactProvenance(BaseModel):
    """Describe the artifact source of a suggester restored from disk."""

    model_config = ConfigDict(extra="forbid")

    artifact_dir: str
    artifact_type: str
    artifact_version: int
    corpus_file: str
    corpus_size: int


class SaytRetrieverSummary(BaseModel):
    """Summarise one configured retriever within a suggester."""

    model_config = ConfigDict(extra="forbid")

    name: str
    spec_type: str
    retriever_type: str
    config: dict[str, Any] = Field(default_factory=dict)
    artifact_provenance: SaytRetrieverArtifactProvenance | None = None


class SaytConfiguration(BaseModel):
    """Return a rich runtime summary of a configured suggester."""

    model_config = ConfigDict(extra="forbid")

    settings: SaytGlobalSettings
    corpus: SaytCorpusSummary
    retrievers: list[SaytRetrieverSummary] = Field(default_factory=list)
    artifact_provenance: SaytArtifactProvenance | None = None


class Suggestion(BaseModel):
    """Represent a SAYT match with its display text and combined score."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    display_text: str
    score: float


def take_with_ties(
    items: list[Suggestion],
    limit: int,
    display_text_value_counts: dict[str, int] | None = None,
) -> list[Suggestion]:
    """Return the first ``limit`` items and any later items tied on score.

    The items are ranked based on the following factors (in order):
        1. Descending score.
        2. Descending display-text duplication count (when provided).
        3. Case-insensitive display-text alphabetical order.
    Duplicates with the same display text are removed, keeping the highest-scoring one.

    Args:
        items: Scored ``Suggestion`` objects to rank.
        limit: Maximum number of leading items before tie extension is applied.
        display_text_value_counts: Optional mapping of display text to occurrence counts
            in the corpus. If provided, items whose display text occurs more than
            once will be considered for higher priority.

    Returns:
        The highest-scoring items up to ``limit``, plus any later items that are
        tied with the cutoff score.
    """
    if limit < 1 or not items:
        return []
    if display_text_value_counts is None:
        display_text_value_counts = {}

    items = sorted(
        items,
        key=lambda kv: (
            -kv.score,
            -display_text_value_counts.get(kv.display_text, 0),
            kv.display_text.lower(),
        ),
    )
    # drop duplicates with the same display text, keeping the highest-scoring one
    seen_display_texts: set[str] = set()
    deduped_items: list[Suggestion] = []
    for item in items:
        if item.display_text not in seen_display_texts:
            deduped_items.append(item)
            seen_display_texts.add(item.display_text)

    if limit >= len(deduped_items):
        return deduped_items

    cutoff_score = float(deduped_items[limit - 1].score)
    end = limit
    while end < len(deduped_items) and float(deduped_items[end].score) == cutoff_score:
        end += 1
    return deduped_items[:end]
