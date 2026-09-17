"""Tests for SAYT configuration validation."""

# pylint: disable=too-few-public-methods, R0801

import pytest

from survey_assist_embed_core.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SAYTBuilder,
    SAYTSuggester,
    SemanticRetrieverSpec,
    default_retriever_specs,
)
from survey_assist_embed_core.sayt.core import CleanCorpus
from survey_assist_embed_core.sayt.weight_specs import (
    NgramWeightSpec,
    PrefixWeightSpec,
    WeightSpecs,
)


@pytest.mark.parametrize(
    "factory, kwargs, exc_type, match",
    [
        (SAYTSuggester, {"min_chars": 2}, ValueError, "min_chars must be >= 3"),
        (
            SAYTSuggester,
            {"min_chars": True},
            TypeError,
            "min_chars must be an integer",
        ),
        (
            SAYTSuggester,
            {"max_suggestions": 0},
            ValueError,
            "max_suggestions must be between 1 and 100",
        ),
        (
            SAYTSuggester,
            {"max_suggestions": 101},
            ValueError,
            "max_suggestions must be between 1 and 100",
        ),
        (
            SAYTSuggester,
            {"min_chars": "abc"},
            TypeError,
            "min_chars must be an integer",
        ),
        (SAYTBuilder, {"min_chars": 2}, ValueError, "min_chars must be >= 3"),
        (
            SAYTBuilder,
            {"min_chars": True},
            TypeError,
            "min_chars must be an integer",
        ),
        (
            SAYTBuilder,
            {"max_suggestions": 0},
            ValueError,
            "max_suggestions must be between 1 and 100",
        ),
        (
            SAYTBuilder,
            {"max_suggestions": 101},
            ValueError,
            "max_suggestions must be between 1 and 100",
        ),
        (
            SAYTBuilder,
            {"max_suggestions": "abc"},
            TypeError,
            "max_suggestions must be an integer",
        ),
    ],
)
def test_runtime_setting_validation(factory, kwargs, exc_type, match):
    """Reject unsupported global SAYT settings on public entry points."""
    with pytest.raises(exc_type, match=match):
        factory([("car wash", "Car Wash")], **kwargs)


def test_default_retriever_specs_returns_standard_set():
    """Provide the standard prefix, n-gram, and semantic specs."""
    specs = default_retriever_specs()

    assert [type(spec).__name__ for spec in specs] == [
        "PrefixRetrieverSpec",
        "NgramRetrieverSpec",
        "SemanticRetrieverSpec",
    ]


@pytest.mark.parametrize(
    "factory, kwargs, match",
    [
        (NgramRetrieverSpec, {"n": 1}, "ngram n must be between 2 and 5"),
        (NgramRetrieverSpec, {"n": 6}, "ngram n must be between 2 and 5"),
        (NgramRetrieverSpec, {"max_df": 0.0}, "ngram max_df must be in"),
        (NgramRetrieverSpec, {"max_df": 1.1}, "ngram max_df must be in"),
        (SemanticRetrieverSpec, {"model": "   "}, "semantic model must be"),
    ],
)
def test_retriever_spec_validation(factory, kwargs, match):
    """Reject invalid retriever-spec settings."""
    with pytest.raises(ValueError, match=match):
        factory(**kwargs)


def test_ngram_retriever_spec_validates_against_corpus_size():
    """Reject n-gram configs that would filter every feature from a corpus."""
    corpus = CleanCorpus.model_validate([("car wash", "Car Wash")])

    with pytest.raises(ValueError, match="ngram max_df is too low"):
        NgramRetrieverSpec(max_df=0.2).build(corpus, min_chars=3)


def test_retriever_specs_keep_their_config():
    """Expose per-retriever settings on the spec object."""
    n = 4
    max_df = 0.8
    spec = NgramRetrieverSpec(n=n, max_df=max_df)

    assert spec.n == n
    assert spec.max_df == pytest.approx(max_df)


def test_suggester_warns_when_weight_names_do_not_match_a_retriever(small_corpus):
    """Warn during construction when retriever and weight names differ."""
    with pytest.warns(RuntimeWarning) as warning_records:
        SAYTSuggester(
            small_corpus,
            min_chars=3,
            retrievers=[PrefixRetrieverSpec()],
            weights=WeightSpecs(specs=[PrefixWeightSpec(), NgramWeightSpec()]),
        )

    warning_messages = [str(record.message) for record in warning_records]
    assert any(
        "Weight specs configured for unknown retrievers: ngram" in message
        for message in warning_messages
    )


def test_update_weights_warns_when_weight_names_do_not_match(small_corpus):
    """Warn when replacing weights with names unknown to the suggester."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[PrefixRetrieverSpec()],
        weights=WeightSpecs(specs=[PrefixWeightSpec()]),
    )

    with pytest.warns(
        RuntimeWarning,
        match="Weight specs configured for unknown retrievers: ngram",
    ):
        suggester.update_weights(
            WeightSpecs(specs=[PrefixWeightSpec(), NgramWeightSpec()])
        )


def test_suggester_warns_when_weight_names_are_duplicate(small_corpus):
    """Capture duplicate weight-spec warnings during construction."""
    with pytest.warns(
        RuntimeWarning,
        match="Duplicate retriever weight specs found for: prefix",
    ):
        SAYTSuggester(
            small_corpus,
            min_chars=3,
            retrievers=[PrefixRetrieverSpec()],
            weights=WeightSpecs(specs=[PrefixWeightSpec(), PrefixWeightSpec()]),
        )


def test_update_weights_warns_when_retriever_weight_is_missing(small_corpus):
    """Capture missing weight-spec warnings during an update."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[PrefixRetrieverSpec(), NgramRetrieverSpec(max_df=1.0)],
        weights=WeightSpecs(specs=[PrefixWeightSpec(), NgramWeightSpec()]),
    )

    with pytest.warns(
        RuntimeWarning,
        match="No weight spec configured for retrievers: ngram",
    ):
        suggester.update_weights(WeightSpecs(specs=[PrefixWeightSpec()]))


def test_semantic_retriever_spec_builds_semantic_retriever(monkeypatch):
    """Delegate semantic retriever construction with the configured settings."""
    corpus = CleanCorpus.model_validate([("car wash", "Car Wash")])
    captured = {}

    class _StubSemanticRetriever:
        def __init__(self, corpus_arg, *, model, vectoriser_class, min_chars):
            captured["corpus"] = corpus_arg
            captured["model"] = model
            captured["vectoriser_class"] = vectoriser_class
            captured["min_chars"] = min_chars

    monkeypatch.setattr(
        "survey_assist_embed_core.sayt.retriever_specs.SemanticRetriever",
        _StubSemanticRetriever,
    )

    spec = SemanticRetrieverSpec(model="custom-model")

    retriever = spec.build(corpus, min_chars=4)

    assert isinstance(retriever, _StubSemanticRetriever)
    assert captured == {
        "corpus": corpus,
        "model": "custom-model",
        "vectoriser_class": None,
        "min_chars": 4,
    }


def test_semantic_retriever_spec_passes_vectoriser_class_to_retriever(monkeypatch):
    """Pass vectoriser_class through to semantic retriever construction."""
    corpus = CleanCorpus.model_validate([("car wash", "Car Wash")])
    captured = {}

    class _StubSemanticRetriever:
        def __init__(self, corpus_arg, *, model, vectoriser_class, min_chars):
            captured["corpus"] = corpus_arg
            captured["model"] = model
            captured["vectoriser_class"] = vectoriser_class
            captured["min_chars"] = min_chars

    monkeypatch.setattr(
        "survey_assist_embed_core.sayt.retriever_specs.SemanticRetriever",
        _StubSemanticRetriever,
    )

    spec = SemanticRetrieverSpec(
        model="custom-model",
        vectoriser_class="OnnxVectoriser",
    )

    retriever = spec.build(corpus, min_chars=4)

    assert isinstance(retriever, _StubSemanticRetriever)
    assert captured == {
        "corpus": corpus,
        "model": "custom-model",
        "vectoriser_class": "OnnxVectoriser",
        "min_chars": 4,
    }
