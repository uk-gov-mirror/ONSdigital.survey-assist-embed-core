"""Shared pytest fixtures for SAYT tests."""

# pylint: disable=redefined-outer-name

import pytest

from survey_assist_embed_core.sayt.retriever_specs import PrefixRetrieverSpec
from survey_assist_embed_core.sayt.suggester import SAYTSuggester
from survey_assist_embed_core.sayt.weight_specs import PrefixWeightSpec, WeightSpecs


@pytest.fixture
def small_corpus():
    """Return a small mixed corpus with duplicates for ranking tests."""
    return [
        ("Car wash", "Car Wash"),
        ("Car wash", "CAR WASH (duplicate)"),
        ("Car waxing", "Car Waxing"),
        ("Waxing car", "Car Waxing"),
        ("Carpentry services", "Carpentry services"),
        ("Dog grooming", "Dog grooming"),
    ]


@pytest.fixture
def prefix_suggester(small_corpus):
    """Return the standard prefix-only suggester used by ranking tests."""
    return SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[PrefixRetrieverSpec()],
        weights=WeightSpecs(specs=[PrefixWeightSpec()]),
    )
