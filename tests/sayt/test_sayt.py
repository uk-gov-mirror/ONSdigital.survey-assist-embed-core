"""Tests for the SAYTSuggester public API."""

# ruff: noqa: PLR2004
# pylint: disable=protected-access,redefined-outer-name,too-few-public-methods,C0116,W0613

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from survey_assist_embed_core.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SAYTBuilder,
    SaytConfiguration,
)
from survey_assist_embed_core.sayt.core import (
    CleanCorpus,
    PersistedCorpusRow,
    Suggestion,
)
from survey_assist_embed_core.sayt.suggester import SAYTSuggester
from survey_assist_embed_core.sayt.weight_specs import (
    NgramWeightSpec,
    PrefixWeightSpec,
    WeightSpecs,
)


@dataclass(frozen=True, slots=True)
class _CustomWeightSpec:
    """Weight spec for the custom retriever configuration tests."""

    weights: float = 1.0
    retriever_name: str = field(init=False, default="custom")

    def get_weight(self, query_length: int | None = None) -> float:
        _ = query_length
        return self.weights


@dataclass(frozen=True, slots=True)
class _SlotsOnlyWeightSpec:
    """Weight spec for the slots-only retriever configuration test."""

    weights: float = 1.0
    retriever_name: str = field(init=False, default="slots-only")

    def get_weight(self, query_length: int | None = None) -> float:
        _ = query_length
        return self.weights


def test_constructor_rejects_unknown_kwargs(small_corpus):
    """Reject unknown constructor kwargs during config validation."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        SAYTSuggester(small_corpus, does_not_exist=True)  # pylint: disable=E1123


def test_empty_corpus_after_filtering_raises():
    """Raise when corpus normalisation removes every input row."""
    corpus = [None, " ", "-9", ("-9", "ignored")]
    with (
        pytest.warns(UserWarning, match="Skipping empty or invalid corpus item"),
        pytest.raises(ValueError, match="corpus is empty"),
    ):
        SAYTSuggester(corpus)


def test_clean_corpus_rows_are_search_display_pairs(small_corpus):
    """Each cleaned row is a (search_text, display_text) pair with no identifier."""
    corpus = CleanCorpus.model_validate(small_corpus)

    assert len(corpus.rows) == len(small_corpus)
    assert all(len(row) == 2 for row in corpus.rows)


def test_clean_corpus_rows_are_sorted_by_search_and_display():
    """Sort cleaned rows by search text, then display text."""
    corpus = CleanCorpus.model_validate(
        [
            ("Dog grooming", "Dog grooming"),
            ("Car wash", "Car Wash"),
            ("Car wash", "CAR WASH (duplicate)"),
            ("Car waxing", "Car Waxing"),
        ]
    )

    assert corpus.rows == [
        ("car wash", "CAR WASH (duplicate)"),
        ("car wash", "Car Wash"),
        ("car waxing", "Car Waxing"),
        ("dog grooming", "Dog grooming"),
    ]


def test_clean_corpus_accepts_existing_instance_and_dict_input(small_corpus):
    """Preserve existing validated input forms through pydantic coercion."""
    corpus = CleanCorpus.model_validate(small_corpus)

    same_corpus = CleanCorpus.model_validate(corpus)
    dict_corpus = CleanCorpus.model_validate({"corpus": small_corpus})

    assert same_corpus.rows == corpus.rows
    assert dict_corpus.rows == corpus.rows


def test_clean_corpus_coerce_input_preserves_models_and_rows_payloads(small_corpus):
    """Return existing CleanCorpus instances and explicit rows payloads unchanged."""
    corpus = CleanCorpus.model_validate(small_corpus)
    rows_payload = {"rows": corpus.rows}

    assert CleanCorpus._coerce_input(corpus) is corpus
    assert CleanCorpus._coerce_input(rows_payload) is rows_payload


def test_clean_corpus_model_dump_excludes_derived_lookup_dicts(small_corpus):
    """Keep derived lookup dictionaries out of the public model fields."""
    corpus = CleanCorpus.model_validate(small_corpus)

    dumped = corpus.model_dump()

    assert "id_to_search" not in dumped
    assert "id_to_display" not in dumped
    assert "display_text_value_counts" not in dumped
    assert dumped["rows"] == corpus.rows


def test_clean_corpus_restores_persisted_rows(small_corpus):
    """Restore cleaned corpus rows exactly as persisted search/display pairs."""
    corpus = CleanCorpus.model_validate(small_corpus)

    restored = CleanCorpus.from_persisted_rows(
        [PersistedCorpusRow(*row) for row in corpus.rows]
    )

    assert restored.rows == corpus.rows
    assert restored.model_dump() == corpus.model_dump()


def test_clean_corpus_rejects_non_iterable_input():
    """Reject scalar corpus values before attempting to clean them."""
    with pytest.raises(TypeError, match="corpus must be an iterable"):
        CleanCorpus._clean_corpus(123)


def test_clean_corpus_warns_and_falls_back_when_display_is_missing():
    """Use the search text when the display value is empty or missing."""
    with pytest.warns(UserWarning, match="using search text as display"):
        corpus = CleanCorpus.model_validate([("Car wash", "")])

    assert corpus.rows[0][1] == "Car wash"


def test_from_csv_builds_and_suggests(tmp_path, small_corpus):
    """Build a suggester from CSV input and return matching suggestions."""
    csv_path = tmp_path / "responses.csv"
    df = pd.DataFrame(
        {
            "search": [x[0] for x in small_corpus],
            "display": [x[1] for x in small_corpus],
        }
    )
    df.to_csv(csv_path, index=False)

    suggester = SAYTSuggester.from_csv(
        str(csv_path),
        search_text_col="search",
        display_text_col="display",
        retrievers=[PrefixRetrieverSpec()],
        min_chars=3,
        max_suggestions=10,
        weights=WeightSpecs(specs=[PrefixWeightSpec()]),
    )

    assert suggester.suggest("car")[0].startswith("Car")


def test_from_csv_uses_search_column_as_default_display(tmp_path, small_corpus):
    """Reuse the search column as display when none is configured."""
    csv_path = tmp_path / "responses.csv"
    pd.DataFrame({"search": [x[0] for x in small_corpus]}).to_csv(csv_path, index=False)

    suggester = SAYTSuggester.from_csv(
        str(csv_path),
        search_text_col="search",
        retrievers=[PrefixRetrieverSpec()],
        min_chars=3,
        weights=WeightSpecs(specs=[PrefixWeightSpec()]),
    )

    assert suggester.suggest("car")[0] == "Car wash"


def test_from_csv_rejects_missing_search_column(tmp_path, small_corpus):
    """Raise when the configured search column is absent from the CSV."""
    csv_path = tmp_path / "responses.csv"
    pd.DataFrame(
        {
            "display": [x[1] for x in small_corpus],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Column 'search' not found"):
        SAYTSuggester.from_csv(str(csv_path), search_text_col="search")


def test_from_csv_rejects_missing_display_column(tmp_path, small_corpus):
    """Raise when the configured display column is absent from the CSV."""
    csv_path = tmp_path / "responses.csv"
    pd.DataFrame(
        {
            "search": [x[0] for x in small_corpus],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Column 'display' not found"):
        SAYTSuggester.from_csv(
            str(csv_path),
            search_text_col="search",
            display_text_col="display",
        )


def test_from_artifact_restores_prefix_suggester(tmp_path, small_corpus):
    """Round-trip a prefix-only artifact into a working suggester."""
    artifact_dir = SAYTBuilder(
        small_corpus,
        retrievers=[PrefixRetrieverSpec()],
        min_chars=3,
        max_suggestions=5,
        weights=WeightSpecs(specs=[PrefixWeightSpec(weights=2.0)]),
    ).build_artifact(tmp_path / "artifact")

    restored = SAYTSuggester.from_artifact(artifact_dir)
    expected = SAYTSuggester(
        small_corpus,
        retrievers=[PrefixRetrieverSpec()],
        min_chars=3,
        max_suggestions=5,
        weights=WeightSpecs(specs=[PrefixWeightSpec(weights=2.0)]),
    )
    restored_config = restored.get_config()
    expected_config = expected.get_config()

    assert restored.suggest("car") == expected.suggest("car")
    assert restored_config.settings == expected_config.settings
    assert restored_config.corpus == expected_config.corpus
    assert [
        retriever.model_dump(exclude={"artifact_provenance"})
        for retriever in restored_config.retrievers
    ] == [
        retriever.model_dump(exclude={"artifact_provenance"})
        for retriever in expected_config.retrievers
    ]
    assert restored_config.artifact_provenance is not None
    assert restored_config.artifact_provenance.artifact_dir == str(artifact_dir)
    assert restored_config.retrievers[0].artifact_provenance is not None
    assert expected_config.artifact_provenance is None


def test_from_artifact_rejects_manifest_corpus_size_mismatch(tmp_path, small_corpus):
    """Reject artifacts whose manifest corpus size disagrees with stored rows."""
    artifact_dir = SAYTBuilder(
        small_corpus,
        retrievers=[PrefixRetrieverSpec()],
        min_chars=3,
        max_suggestions=5,
        weights=WeightSpecs(specs=[PrefixWeightSpec(weights=1.0)]),
    ).build_artifact(tmp_path / "artifact")
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["corpus_size"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError, match="Artifact corpus size does not match manifest"
    ):
        SAYTSuggester.from_artifact(artifact_dir)


def test_get_config_returns_rich_runtime_summary(small_corpus):
    """Expose runtime settings, corpus stats, and retriever summaries."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        max_suggestions=5,
        retrievers=[
            PrefixRetrieverSpec(),
            NgramRetrieverSpec(n=4, max_df=1.0),
        ],
        weights=WeightSpecs(
            specs=[PrefixWeightSpec(weights=2.0), NgramWeightSpec(weights=1.0)]
        ),
    )

    config = suggester.get_config()
    display_counts = Counter(display for _, display in suggester._corpus.rows)

    assert isinstance(config, SaytConfiguration)
    assert config.settings.model_dump() == {
        "min_chars": 3,
        "max_suggestions": 5,
    }
    assert config.corpus.model_dump() == {
        "size": suggester._corpus.size,
        "unique_display_texts": len(display_counts),
        "max_duplication": max(display_counts.values(), default=0),
    }
    assert [retriever.name for retriever in config.retrievers] == ["prefix", "ngram"]
    assert config.retrievers[0].config == {}
    assert config.retrievers[1].config == {"n": 4, "max_df": 1.0}
    assert config.retrievers[1].retriever_type == "NgramRetriever"
    assert [spec.model_dump() for spec in config.weight_specs.specs] == [
        {
            "retriever_name": "prefix",
            "spec_type": "PrefixWeightSpec",
            "weights": 2.0,
            "normalised_weights": pytest.approx(2 / 3),
        },
        {
            "retriever_name": "ngram",
            "spec_type": "NgramWeightSpec",
            "weights": 1.0,
            "normalised_weights": pytest.approx(1 / 3),
        },
    ]
    assert config.artifact_provenance is None


def test_get_config_supports_custom_specs_without_artifact_handlers(small_corpus):
    """Summarise custom runtime-only specs without requiring persistence hooks."""

    class _StubRetriever:
        def suggest_with_scores(self, q_norm, num_suggestions):
            _ = (q_norm, num_suggestions)
            return []

    class _CustomSpec:
        def __init__(self, *, trigger: str):
            self.trigger = trigger
            self.name = "custom"

        def build(self, corpus, *, min_chars):
            _ = (corpus, min_chars)
            return _StubRetriever()

    config = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[_CustomSpec(trigger="groom")],
        weights=WeightSpecs(specs=[_CustomWeightSpec()]),
    ).get_config()

    assert config.retrievers[0].config == {"trigger": "groom"}
    assert config.retrievers[0].artifact_provenance is None


def test_get_config_serialises_nested_custom_spec_values(small_corpus):
    """Convert non-JSON-native custom spec config values into JSON-safe forms."""

    class _StubRetriever:
        def suggest_with_scores(self, q_norm, num_suggestions):
            _ = (q_norm, num_suggestions)
            return []

    class _Marker:
        def __str__(self):
            return "marker-object"

    class _CustomSpec:
        def __init__(self):
            self.name = "custom"
            self.folder = Path("artifacts/model")
            self.options = {
                "labels": ["car", Path("cache/index"), _Marker()],
                "metadata": {"marker": _Marker()},
            }
            self.values = (Path("weights.bin"), _Marker())

        def build(self, corpus, *, min_chars):
            _ = (corpus, min_chars)
            return _StubRetriever()

    config = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[_CustomSpec()],
        weights=WeightSpecs(specs=[_CustomWeightSpec()]),
    ).get_config()

    assert config.retrievers[0].config == {
        "folder": "artifacts/model",
        "options": {
            "labels": ["car", "cache/index", "marker-object"],
            "metadata": {"marker": "marker-object"},
        },
        "values": ["weights.bin", "marker-object"],
    }


def test_get_config_returns_empty_config_for_slots_only_custom_spec(small_corpus):
    """Return an empty config summary when a custom spec exposes no __dict__."""

    class _StubRetriever:
        def suggest_with_scores(self, q_norm, num_suggestions):
            _ = (q_norm, num_suggestions)
            return []

    class _SlotsOnlySpec:
        __slots__ = ("name", "trigger")

        def __init__(self, *, trigger: str):
            self.name = "slots-only"
            self.trigger = trigger

        def build(self, corpus, *, min_chars):
            _ = (corpus, min_chars)
            return _StubRetriever()

    config = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[_SlotsOnlySpec(trigger="groom")],
        weights=WeightSpecs(specs=[_SlotsOnlyWeightSpec()]),
    ).get_config()

    assert config.retrievers[0].config == {}


def test_suggest_returns_empty_for_short_or_non_string_query(small_corpus):
    """Return no suggestions for short or non-string queries."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=4,
        retrievers=[PrefixRetrieverSpec()],
        weights=WeightSpecs(specs=[PrefixWeightSpec()]),
    )
    assert not suggester.suggest("car")
    assert not suggester.suggest(None)


def test_suggest_with_scores_defaults_to_config_max_suggestions(small_corpus):
    """Use the configured max_suggestions, but keep ties at the cutoff."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        max_suggestions=2,
        retrievers=[PrefixRetrieverSpec()],
        weights=WeightSpecs(specs=[PrefixWeightSpec()]),
    )

    results = suggester.suggest_with_scores("car")

    assert {result.display_text for result in results} == {
        "Car Waxing",
        "Car Wash",
        "CAR WASH (duplicate)",
        "Carpentry services",
    }


def test_suggest_respects_explicit_num_suggestions(small_corpus):
    """Allow callers to override the configured limit, while keeping ties."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        max_suggestions=5,
        retrievers=[PrefixRetrieverSpec()],
        weights=WeightSpecs(specs=[PrefixWeightSpec()]),
    )

    assert len(suggester.suggest("car", num_suggestions=1)) == 4


def test_suggest_with_scores_keeps_ties_at_cutoff(prefix_suggester):
    """Keep all tied scored suggestions at the public cutoff."""
    results = prefix_suggester.suggest_with_scores("car", num_suggestions=1)

    assert len({result.display_text for result in results}) == 4


def test_suggest_keeps_ties_at_cutoff(prefix_suggester):
    """Keep all tied display suggestions at the public cutoff."""
    results = prefix_suggester.suggest("car", num_suggestions=1)

    assert results == [
        "Car Waxing",
        "Car Wash",
        "CAR WASH (duplicate)",
        "Carpentry services",
    ]


def test_suggest_with_scores_uses_only_supplied_retrievers(small_corpus):
    """Delegate only to the configured retriever specs."""
    semantic_calls = []

    class _StubRetriever:
        def __init__(self, row):
            self._row = row

        def suggest_with_scores(self, q_norm, num_suggestions):
            semantic_calls.append((q_norm, num_suggestions))
            return [Suggestion(display_text=self._row[1], score=3.0)]

    @dataclass(frozen=True, slots=True)
    class _StubRetrieverSpec:
        name: str = "stub"

        def build(self, corpus, *, min_chars):
            return _StubRetriever(corpus.rows[0])

    @dataclass(frozen=True, slots=True)
    class _StubWeightSpec:
        weights: float = 1.0
        retriever_name: str = "stub"

        def get_weight(self, query_length: int | None = None) -> float:
            return self.weights

    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[_StubRetrieverSpec()],
        weights=WeightSpecs(specs=[_StubWeightSpec()]),
    )

    results = suggester.suggest_with_scores("car")

    assert semantic_calls == [("car", 50)]
    assert [result.display_text for result in results] == [suggester._corpus.rows[0][1]]


def test_suggest_with_scores_applies_per_call_weight_override(small_corpus):
    """Apply valid query-specific weight overrides without rebuilding retrievers."""

    class _StubRetriever:
        def __init__(self, display_text):
            self.display_text = display_text

        def suggest_with_scores(self, q_norm, num_suggestions):
            _ = (q_norm, num_suggestions)
            return [Suggestion(display_text=self.display_text, score=1.0)]

    @dataclass(frozen=True, slots=True)
    class _StubRetrieverSpec:
        name: str
        display_text: str

        def build(self, corpus, *, min_chars):
            _ = (corpus, min_chars)
            return _StubRetriever(self.display_text)

    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[
            _StubRetrieverSpec(name="prefix", display_text="First"),
            _StubRetrieverSpec(name="ngram", display_text="Second"),
        ],
        weights=WeightSpecs(
            specs=[
                PrefixWeightSpec(weights=1.0),
                NgramWeightSpec(weights=1.0),
            ]
        ),
    )

    results = suggester.suggest_with_scores(
        "car",
        weights=WeightSpecs(
            specs=[
                PrefixWeightSpec(weights=3.0),
                NgramWeightSpec(weights=1.0),
            ]
        ),
    )

    assert [result.display_text for result in results] == ["First", "Second"]
    assert [result.score for result in results] == pytest.approx([0.75, 0.25])


def test_update_weights_changes_public_scores(small_corpus):
    """Apply replacement weights to subsequent suggestions."""

    class _StubRetriever:
        def __init__(self, display_text):
            self.display_text = display_text

        def suggest_with_scores(self, q_norm, num_suggestions):
            _ = (q_norm, num_suggestions)
            return [Suggestion(display_text=self.display_text, score=1.0)]

    @dataclass(frozen=True, slots=True)
    class _StubRetrieverSpec:
        name: str
        display_text: str

        def build(self, corpus, *, min_chars):
            _ = (corpus, min_chars)
            return _StubRetriever(self.display_text)

    def _weights(first: float, second: float) -> WeightSpecs:
        return WeightSpecs(
            specs=[
                PrefixWeightSpec(weights=first),
                NgramWeightSpec(weights=second),
            ]
        )

    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[
            _StubRetrieverSpec(name="prefix", display_text="First"),
            _StubRetrieverSpec(name="ngram", display_text="Second"),
        ],
        weights=_weights(1.0, 1.0),
    )

    suggester.update_weights(_weights(1.0, 3.0))

    results = suggester.suggest_with_scores("car")

    assert [result.display_text for result in results] == ["Second", "First"]
    assert [result.score for result in results] == pytest.approx([0.75, 0.25])


def test_zero_weight_excludes_retriever_from_public_results(small_corpus):
    """Do not query or include a retriever whose configured weight is zero."""
    calls = []

    class _StubRetriever:
        def __init__(self, name):
            self.name = name

        def suggest_with_scores(self, q_norm, num_suggestions):
            _ = (q_norm, num_suggestions)
            calls.append(self.name)
            return [Suggestion(display_text=self.name, score=1.0)]

    @dataclass(frozen=True, slots=True)
    class _StubRetrieverSpec:
        name: str

        def build(self, corpus, *, min_chars):
            _ = (corpus, min_chars)
            return _StubRetriever(self.name)

    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[
            _StubRetrieverSpec(name="prefix"),
            _StubRetrieverSpec(name="ngram"),
        ],
        weights=WeightSpecs(
            specs=[
                PrefixWeightSpec(weights=1.0),
                NgramWeightSpec(weights=0.0),
            ]
        ),
    )

    assert suggester.suggest("car") == ["prefix"]
    assert calls == ["prefix"]


def test_per_call_empty_weights_are_rejected(prefix_suggester):
    """Reject an empty per-call weight override."""
    with pytest.raises(ValueError, match="At least one retriever weight"):
        prefix_suggester.suggest("car", weights=WeightSpecs(specs=[]))


def test_per_call_zero_total_weights_are_rejected(prefix_suggester):
    """Reject a per-call override whose total weight is zero."""
    with pytest.raises(ValueError, match="Total weight cannot be zero"):
        prefix_suggester.suggest(
            "car",
            weights=WeightSpecs(specs=[PrefixWeightSpec(weights=0.0)]),
        )


def test_update_weights_rejects_zero_total_without_replacing_current_weights(
    prefix_suggester,
):
    """Keep the active weights when an invalid update is rejected."""
    with pytest.raises(ValueError, match="Total weight cannot be zero"):
        prefix_suggester.update_weights(
            WeightSpecs(specs=[PrefixWeightSpec(weights=0.0)])
        )

    assert prefix_suggester.suggest("car")


def test_suggestion_model_dump_is_api_friendly() -> None:
    """Expose a simple serialisable payload for endpoint responses."""
    suggestion = Suggestion(display_text="Car Wash", score=0.75)

    assert suggestion.model_dump() == {
        "display_text": "Car Wash",
        "score": 0.75,
    }


def test_combine_suggestions_ignores_non_positive_score_groups(prefix_suggester):
    """Drop a retriever group entirely when its max score is not positive."""
    combined = prefix_suggester._combine_suggestions(
        [
            (
                1.0,
                [
                    Suggestion(
                        display_text=prefix_suggester._corpus.rows[0][1],
                        score=0.0,
                    )
                ],
            ),
            (1.0, []),
            (1.0, []),
        ]
    )

    assert combined == []


def test_combine_suggestions_ignores_invalid_scores(prefix_suggester):
    """Ignore missing row ids and keep distinct row ids in combined scores."""
    first_display = prefix_suggester._corpus.rows[0][1]
    second_display = prefix_suggester._corpus.rows[2][1]

    combined = prefix_suggester._combine_suggestions(
        [
            (1.0, [Suggestion(display_text=first_display, score=0.0)]),
            (
                1.0,
                [
                    Suggestion(display_text=first_display, score=2.0),
                    Suggestion(display_text=second_display, score=1.0),
                ],
            ),
        ]
    )

    assert combined == [
        Suggestion(display_text=first_display, score=1.0),
        Suggestion(display_text=second_display, score=0.5),
    ]


def test_suggester_defaults_to_standard_retriever_specs(monkeypatch, small_corpus):
    """Use the standard prefix, n-gram, and semantic specs when none are supplied."""

    class _StubRetriever:
        def suggest_with_scores(self, q_norm, num_suggestions):
            return []

    @dataclass(frozen=True, slots=True)
    class _StubRetrieverSpec:
        name: str

        def build(self, corpus, *, min_chars):
            return _StubRetriever()

    monkeypatch.setattr(
        "survey_assist_embed_core.sayt._base.default_retriever_specs",
        lambda: [
            _StubRetrieverSpec(name="prefix"),
            _StubRetrieverSpec(name="ngram"),
            _StubRetrieverSpec(name="semantic"),
        ],
    )

    suggester = SAYTSuggester(small_corpus, min_chars=3)

    assert [configured.name for configured in suggester._retrievers] == [
        "prefix",
        "ngram",
        "semantic",
    ]


def test_constructor_rejects_empty_retriever_list(small_corpus):
    """Reject suggester construction without any retriever specs."""
    with pytest.raises(ValueError, match="At least one retriever"):
        SAYTSuggester(small_corpus, retrievers=[], weights=WeightSpecs(specs=[]))


def test_constructor_rejects_invalid_custom_retriever_weight(small_corpus):
    """Reject custom retriever specs whose own weight is invalid."""
    build_calls = []

    class _StubRetriever:
        def suggest_with_scores(self, q_norm, num_suggestions):
            return []

    @dataclass(frozen=True, slots=True)
    class _StubRetrieverSpec:
        name: str = "stub"
        weight: float = 1.0

        def build(self, corpus, *, min_chars):
            build_calls.append((corpus, min_chars))
            return _StubRetriever()

    @dataclass(frozen=True, slots=True)
    class _NegativeStubRetrieverSpec:
        name: str = "negative"
        weight: float = -0.5

        def build(self, corpus, *, min_chars):
            build_calls.append((corpus, min_chars))
            return _StubRetriever()

    @dataclass(frozen=True, slots=True)
    class _NanStubRetrieverSpec:
        name: str = "nan"
        weight: float = float("nan")

        def build(self, corpus, *, min_chars):
            build_calls.append((corpus, min_chars))
            return _StubRetriever()

    @dataclass(frozen=True, slots=True)
    class _NamedWeightSpec:
        retriever_name: str
        weights: float = 1.0

        def get_weight(self, query_length: int | None = None) -> float:
            _ = query_length
            return self.weights

    with pytest.raises(
        ValueError,
        match="Retriever 'negative' weight must be a finite value => 0",
    ):
        SAYTSuggester(
            small_corpus,
            retrievers=[_StubRetrieverSpec(), _NegativeStubRetrieverSpec()],
            weights=WeightSpecs(
                specs=[
                    _NamedWeightSpec("stub"),
                    _NamedWeightSpec("negative", weights=-1),
                ]
            ),
        )

    with pytest.raises(
        ValueError,
        match="Retriever 'nan' weight must be a finite value => 0",
    ):
        SAYTSuggester(
            small_corpus,
            retrievers=[_StubRetrieverSpec(), _NanStubRetrieverSpec()],
            weights=WeightSpecs(
                specs=[
                    _NamedWeightSpec("stub"),
                    _NamedWeightSpec("nan", weights=np.nan),
                ]
            ),
        )

    assert not build_calls


def test_clean_corpus_rejects_empty_persisted_rows():
    """Reject empty persisted row collections during artifact restore."""
    with pytest.raises(ValueError, match="corpus is empty after filtering"):
        CleanCorpus.from_persisted_rows([])


def test_clean_corpus_coerces_persisted_tuple_values_to_strings():
    """Coerce tuple-based persisted rows to strings before rebuilding indexes."""
    restored = CleanCorpus.from_persisted_rows([(123, 456)])

    assert restored.rows == [("123", "456")]
