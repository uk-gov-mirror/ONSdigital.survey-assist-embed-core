"""Run a small interactive example for industry/organisation description SAYT."""

# pylint: disable=protected-access, R0801

# %%
from survey_assist_embed_core.sayt import (
    NgramRetrieverSpec,
    NgramWeightSpec,
    PrefixRetrieverSpec,
    PrefixWeightSpec,
    SAYTSuggester,
    SemanticRetrieverSpec,
    SemanticWeightSpec,
    WeightSpecs,
)
from survey_assist_embed_core.sayt.core import _normalise

# %%
############# toy example to verify SAYT works #############
small_corpus = [
    ("Car wash", "Car Wash"),
    ("Car wash", "CAR WASH (duplicate)"),
    ("Car waxing", "Car Waxing"),
    ("Waxing car", "Car Waxing"),
    ("Carpentry services", "Carpentry services"),
    ("Dog grooming", "Dog grooming"),
    ("Cat grooming", "Cat grooming"),
    ("USed car sales", "Used car sales"),
    ("Car rental", "Car rental"),
    ("Car repair", "Car repair"),
    ("Car servicing", "Car servicing"),
]

# Define weights
weights: WeightSpecs = WeightSpecs(
    specs=[
        PrefixWeightSpec(),
        NgramWeightSpec(weights=2),
        SemanticWeightSpec(weights={4: 2, 6: 3.0}),
    ]
)

# set max_df high to avoid filtering out n-grams in this tiny corpus
suggester = SAYTSuggester(
    small_corpus,
    retrievers=[
        PrefixRetrieverSpec(),
        NgramRetrieverSpec(max_df=0.8),
        SemanticRetrieverSpec(),
    ],
    weights=weights,
)


# %%
for query in ["car", "cars", "waxi", "grom", "wash", "duplicate", "auto"]:
    # We would not normally call the retrievers directly like this, but it is
    # useful to verify they are wired up as expected and to see their
    # individual contributions before we look at the combined suggestions.
    query_norm = _normalise(query)
    print("searching for:", query)
    for configured in suggester._retrievers:
        print(
            configured.name,
            "->",
            [
                suggestion.model_dump()
                for suggestion in configured.retriever.suggest_with_scores(
                    query_norm, 5
                )
            ],
        )
    print(
        "combined",
        "->",
        [
            suggestion.model_dump()
            for suggestion in suggester.suggest_with_scores(query, 5)
        ],
    )
    print("combined_nice", "->", suggester.suggest(query, 5))
    print()

# %%
