"""Tests for SAYT weight specifications."""

import math
import warnings

import pytest

from survey_assist_embed_core.sayt.retriever_specs import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SemanticRetrieverSpec,
    default_retriever_specs,
)
from survey_assist_embed_core.sayt.weight_specs import (
    NgramWeightSpec,
    PrefixWeightSpec,
    SemanticWeightSpec,
    WeightSpecs,
    default_weight_specs,
)


def test_weight_specs_have_expected_retriever_names_and_defaults():
    """Expose the standard names and default weight values."""
    specs = default_weight_specs()

    assert [spec.retriever_name for spec in specs.specs] == [
        "prefix",
        "ngram",
        "semantic",
    ]
    assert [spec.weights for spec in specs.specs] == [1.0, 1.0, 1.0]


@pytest.mark.parametrize(
    "spec_factory, expected_name",
    [
        (PrefixWeightSpec, "prefix"),
        (NgramWeightSpec, "ngram"),
        (SemanticWeightSpec, "semantic"),
    ],
)
def test_weight_spec_subclasses_set_retriever_name(spec_factory, expected_name):
    """Identify the retriever represented by each weight spec."""
    assert spec_factory().retriever_name == expected_name


@pytest.mark.parametrize(
    "retriever_spec_factory, weight_spec_factory",
    [
        (PrefixRetrieverSpec, PrefixWeightSpec),
        (NgramRetrieverSpec, NgramWeightSpec),
        (SemanticRetrieverSpec, SemanticWeightSpec),
    ],
)
def test_retriever_and_weight_spec_names_match(
    retriever_spec_factory, weight_spec_factory
):
    """Keep each retriever spec aligned with its weight spec."""
    assert retriever_spec_factory().name == weight_spec_factory().retriever_name


def test_default_retriever_and_weight_spec_names_match():
    """Keep the default retriever and weight-spec collections aligned."""
    retriever_names = {spec.name for spec in default_retriever_specs()}
    weight_names = {spec.retriever_name for spec in default_weight_specs().specs}

    assert retriever_names == weight_names


def test_each_default_retriever_has_a_weight_spec_lookup():
    """Provide a weight spec for every default retriever name."""
    weight_specs = default_weight_specs()

    for retriever_spec in default_retriever_specs():
        weight_spec = weight_specs.get_weight_spec(retriever_spec.name)

        assert weight_spec is not None
        assert weight_spec.retriever_name == retriever_spec.name


@pytest.mark.parametrize(
    "weight_specs, retriever_names, warning_match",
    [
        (
            WeightSpecs(specs=[PrefixWeightSpec()]),
            ["prefix", "ngram"],
            "No weight spec configured for retrievers: ngram",
        ),
        (
            WeightSpecs(specs=[PrefixWeightSpec(), NgramWeightSpec()]),
            ["prefix"],
            "Weight specs configured for unknown retrievers: ngram",
        ),
        (
            WeightSpecs(specs=[PrefixWeightSpec(), PrefixWeightSpec()]),
            ["prefix"],
            "Duplicate retriever weight specs found for: prefix",
        ),
    ],
)
def test_weight_specs_warn_about_name_mismatches(
    weight_specs, retriever_names, warning_match
):
    """Warn about incomplete, extra, or duplicate weight-spec names."""
    with pytest.warns(RuntimeWarning, match=warning_match):
        weight_specs.warn_for_retriever_names(retriever_names)


def test_weight_specs_do_not_warn_when_names_match():
    """Keep aligned retriever and weight-spec names warning-free."""
    weight_specs = WeightSpecs(specs=[PrefixWeightSpec(), NgramWeightSpec()])

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        weight_specs.warn_for_retriever_names(["prefix", "ngram"])

    assert [
        warning for warning in caught_warnings if warning.category is RuntimeWarning
    ] == []


def test_weight_specs_report_all_name_mismatches():
    """Report duplicate, missing, and extra names together."""
    weight_specs = WeightSpecs(
        specs=[PrefixWeightSpec(), PrefixWeightSpec(), NgramWeightSpec()]
    )

    with pytest.warns(RuntimeWarning) as warning_records:
        weight_specs.warn_for_retriever_names(["semantic"])

    warning_messages = [str(record.message) for record in warning_records]
    assert any(
        "Duplicate retriever weight specs found for: prefix" in message
        for message in warning_messages
    )
    assert any(
        "No weight spec configured for retrievers: semantic" in message
        for message in warning_messages
    )
    assert any(
        "Weight specs configured for unknown retrievers: ngram, prefix" in message
        for message in warning_messages
    )


@pytest.mark.parametrize("weight", [-1, float("inf"), float("-inf"), math.nan])
def test_weight_config_rejects_invalid_fixed_weights(weight):
    """Reject negative and non-finite fixed weights."""
    with pytest.raises(ValueError, match="finite value >= 0"):
        PrefixWeightSpec(weights=weight)


@pytest.mark.parametrize("length", [0, -1, 1.5, "2"])
def test_weight_config_rejects_invalid_query_lengths(length):
    """Require positive integer query-length keys."""
    with pytest.raises(ValueError, match="positive int"):
        PrefixWeightSpec(weights={length: 1.0})


def test_weight_config_rejects_invalid_dict_weight_values():
    """Reject invalid values in query-length-specific weights."""
    with pytest.raises(ValueError, match="finite value >= 0"):
        PrefixWeightSpec(weights={1: -0.1})


def test_fixed_weight_is_returned_for_any_query_length():
    """Use a fixed weight regardless of query length."""
    spec = PrefixWeightSpec(weights=2.5)

    assert spec.get_weight() == pytest.approx(2.5)
    assert spec.get_weight(1) == pytest.approx(2.5)
    assert spec.get_weight(20) == pytest.approx(2.5)


def test_integer_fixed_weight_is_supported():
    """Accept integer values anywhere a fixed weight is expected."""
    integer_weight = 2
    spec = PrefixWeightSpec(weights=integer_weight)

    assert spec.weights == integer_weight
    assert spec.get_weight(4) == pytest.approx(float(integer_weight))


def test_dict_weight_uses_nearest_lower_query_length():
    """Use the most specific configured weight not exceeding the query length."""
    spec = PrefixWeightSpec(weights={1: 0.5, 4: 1.0, 8: 2.0})

    assert spec.get_weight(1) == pytest.approx(0.5)
    assert spec.get_weight(7) == pytest.approx(1.0)
    assert spec.get_weight(8) == pytest.approx(2.0)


def test_dict_weight_returns_zero_before_first_configured_length():
    """Return zero when no configured query length applies."""
    spec = PrefixWeightSpec(weights={3: 1.0})

    assert spec.get_weight(2) == 0.0


def test_empty_dict_weight_returns_zero():
    """Return zero when no query-length-specific weights are configured."""
    spec = PrefixWeightSpec(weights={})

    assert spec.get_weight(1) == 0.0


def test_dict_weight_requires_query_length():
    """Require a query length when weights vary by query length."""
    with pytest.raises(ValueError, match="query_length must be provided"):
        PrefixWeightSpec(weights={1: 1.0}).get_weight()


def test_weight_specs_find_weights_by_retriever_name():
    """Retrieve a configured weight spec or return None when absent."""
    specs = WeightSpecs(specs=[PrefixWeightSpec(weights=2.0)])

    assert specs.get_weight_spec("prefix") == PrefixWeightSpec(weights=2.0)
    assert specs.get_weight_spec("ngram") is None


def test_weight_specs_get_weight_returns_zero_for_unknown_retriever():
    """Return zero when no weight spec exists for a retriever."""
    specs = WeightSpecs(specs=[PrefixWeightSpec(weights=2.0)])

    assert specs.get_weight("missing") == 0.0


def test_fixed_weights_are_normalised_and_zero_weights_are_omitted():
    """Normalise fixed weights and omit retrievers with zero weight."""
    specs = WeightSpecs(
        specs=[
            PrefixWeightSpec(weights=1.0),
            NgramWeightSpec(weights=3.0),
            SemanticWeightSpec(weights=0.0),
        ]
    )

    assert specs.get_normalised_weights() == {
        "prefix": pytest.approx(0.25),
        "ngram": pytest.approx(0.75),
    }


def test_mixed_weights_include_query_length_one_for_fixed_weights():
    """Give fixed weights a length-one entry when mixed with dictionary weights."""
    specs = WeightSpecs(
        specs=[
            PrefixWeightSpec(weights=1.0),
            NgramWeightSpec(weights={3: 3.0}),
        ]
    )

    assert specs.get_normalised_weights() == {
        "prefix": {1: pytest.approx(1.0), 3: pytest.approx(0.25)},
        "ngram": {1: pytest.approx(0.0), 3: pytest.approx(0.75)},
    }


def test_weights_by_query_length_gives_correct_dict():
    """Return the correct dictionary of weights for each query length."""
    specs = WeightSpecs(
        specs=[
            PrefixWeightSpec(weights={1: 1.0, 2: 2.0, 5: 4.0}),
            NgramWeightSpec(weights={1: 3.0, 3: 1.0}),
        ]
    )

    assert specs.get_normalised_weights() == {
        "prefix": {
            1: pytest.approx(0.25),
            2: pytest.approx(0.4),
            3: pytest.approx(0.6666666),
            5: pytest.approx(0.8),
        },
        "ngram": {
            1: pytest.approx(0.75),
            2: pytest.approx(0.6),
            3: pytest.approx(0.3333333),
            5: pytest.approx(0.2),
        },
    }


def test_query_length_limits_normalisation_to_requested_length():
    """Normalise only the requested query length when one is supplied."""
    specs = WeightSpecs(
        specs=[
            PrefixWeightSpec(weights={1: 1.0, 5: 2.0}),
            NgramWeightSpec(weights={1: 3.0, 5: 2.0}),
        ]
    )

    assert specs.get_normalised_weights(query_length=5) == {
        "prefix": {5: pytest.approx(0.5)},
        "ngram": {5: pytest.approx(0.5)},
    }


def test_requested_query_length_adds_length_one_for_mixed_weights_only_when_omitted():
    """Use only the requested length when normalising mixed weights."""
    specs = WeightSpecs(
        specs=[
            PrefixWeightSpec(weights=1.0),
            NgramWeightSpec(weights={3: 3.0}),
        ]
    )

    assert specs.get_normalised_weights(query_length=3) == {
        "prefix": {3: pytest.approx(0.25)},
        "ngram": {3: pytest.approx(0.75)},
    }


def test_normalisation_rejects_non_positive_query_length_state_in_weight_config():
    """Reject malformed query-length state during normalisation."""
    weights = {1: 1.0}
    spec = PrefixWeightSpec(weights=weights)
    weights[0] = 1.0

    with pytest.raises(
        ValueError,
        match="Query length in weight config must be a positive int, got 0",
    ):
        WeightSpecs(specs=[spec, NgramWeightSpec(weights=1.0)]).get_normalised_weights()


def test_zero_weight_is_retained_in_query_length_normalisation():
    """Represent a configured zero weight explicitly in length-based output."""
    specs = WeightSpecs(
        specs=[
            PrefixWeightSpec(weights={1: 0.0, 3: 1.0}),
            NgramWeightSpec(weights={1: 2.0, 3: 0.0}),
        ]
    )

    assert specs.get_normalised_weights() == {
        "prefix": {1: pytest.approx(0.0), 3: pytest.approx(1.0)},
        "ngram": {1: pytest.approx(1.0), 3: pytest.approx(0.0)},
    }


@pytest.mark.parametrize(
    "specs, query_length",
    [
        (WeightSpecs(specs=[]), None),
        (WeightSpecs(specs=[PrefixWeightSpec(weights=0.0)]), None),
        (WeightSpecs(specs=[PrefixWeightSpec(weights={1: 0.0})]), None),
    ],
)
def test_normalisation_rejects_zero_total_weight(specs, query_length):
    """Reject configurations that cannot produce a positive total weight."""
    with pytest.raises(ValueError, match="Total weight cannot be zero"):
        specs.get_normalised_weights(query_length=query_length)
