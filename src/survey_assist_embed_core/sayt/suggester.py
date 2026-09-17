"""Search-as-you-type (SAYT) orchestration.

This module provides the public suggester API that coordinates configured
retrievers and combines their scores into ranked suggestions.
"""

import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

from survey_assist_utils.logging import get_logger

from survey_assist_embed_core.sayt._base import BaseCorpusBound
from survey_assist_embed_core.sayt.core import (
    CleanCorpus,
    SaytArtifactProvenance,
    SaytConfiguration,
    SaytCorpusSummary,
    SaytGlobalSettings,
    SaytRetrieverArtifactProvenance,
    SaytRetrieverSummary,
    Suggestion,
    _normalise,
    take_with_ties,
)
from survey_assist_embed_core.sayt.retriever_specs import Retriever, RetrieverSpec
from survey_assist_embed_core.sayt.storage import (
    SAYT_ARTIFACT_TYPE,
    SAYT_ARTIFACT_VERSION,
    StoredRetrieverSpec,
    load_retriever_from_artifact,
    read_artifact_corpus,
    read_artifact_manifest,
)
from survey_assist_embed_core.sayt.weight_specs import WeightSpecs

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _ConfiguredRetriever:
    """Runtime retriever."""

    name: str
    retriever: Retriever


class SAYTSuggester(BaseCorpusBound):  # pylint: disable=too-many-instance-attributes
    """Suggest free-text responses as a user types.

    The suggester:
    - validates and cleans the supplied corpus
    - builds the configured retrievers for that corpus
    - combines retriever-local scores into a shared weighted ranking

    By default it uses the standard prefix, n-gram, and semantic retriever
    specifications. Use ``retrievers=`` to override that mix.

    Suggester-wide settings are configured directly on the suggester. At
    present these include:
    - ``min_chars``: minimum query length before retrieval runs
    - ``max_suggestions``: default maximum number of ranked suggestions to return

    Examples:
        Basic usage with an in-memory corpus:

            ```python
            from survey_assist_embed_core.sayt import SAYTSuggester

            suggester = SAYTSuggester(
                corpus=[
                    ("Car wash", "Car Wash"),
                    ("Dog grooming", "Dog grooming"),
                ],
                min_chars=3,
                max_suggestions=5,
            )

            results = suggester.suggest("car")
            ```

        Usage with custom retriever specifications:

            ```python
            from survey_assist_embed_core.sayt import (
                PrefixRetrieverSpec,
                SAYTSuggester,
            )

            suggester = SAYTSuggester(
                corpus=[("Car wash", "Car Wash")],
                retrievers=[PrefixRetrieverSpec()],
                min_chars=3,
            )
            ```
    """

    def __init__(
        self,
        corpus: Iterable[tuple[object, object]] | Iterable[str],
        *,
        retrievers: Sequence[RetrieverSpec] | None = None,
        weights: WeightSpecs | None = None,
        min_chars: int = 4,
        max_suggestions: int = 10,
    ) -> None:
        """Initialise a suggester for a cleaned response corpus.

        Args:
            corpus: Iterable of search strings or ``(search_text, display_text)``
                pairs.
            retrievers: Optional retriever specifications. When omitted, the
                standard prefix, n-gram, and semantic spec set is used.
            weights: Optional retriever weight specifications. When omitted, the
                standard prefix, n-gram, and semantic weight set is used.
            min_chars: Minimum query length before retrieval runs.
            max_suggestions: Default maximum number of ranked suggestions to
                return.
        """
        super().__init__(
            corpus,
            retrievers=retrievers,
            weights=weights,
            min_chars=min_chars,
            max_suggestions=max_suggestions,
        )
        self._weight_specs.warn_for_retriever_names(
            [spec.name for spec in self._retriever_specs]
        )
        self._retrievers = self._build_retrievers(self._retriever_specs)
        self._stored_retrievers: tuple[StoredRetrieverSpec, ...] | None = None
        self._artifact_provenance: SaytArtifactProvenance | None = None
        self._weights = _normalised_weight_specs(self._weight_specs)

        logger.info(
            "SAYT suggester initialised",
            corpus_size=self._corpus.size,
            retriever_count=len(self._retrievers),
            min_chars=self._min_chars,
            max_suggestions=self._max_suggestions,
        )

    @classmethod
    def _from_state(  # pylint: disable=too-many-arguments  # noqa: PLR0913
        cls,
        *,
        corpus: CleanCorpus,
        min_chars: int,
        max_suggestions: int,
        retriever_specs: Sequence[RetrieverSpec],
        retrievers: list[_ConfiguredRetriever],
        stored_retrievers: Sequence[StoredRetrieverSpec] | None = None,
        weight_specs: WeightSpecs,
        weights: WeightSpecs,
        artifact_provenance: SaytArtifactProvenance | None = None,
    ) -> "SAYTSuggester":
        """Construct a suggester from already-validated runtime state."""
        suggester = cls.__new__(cls)
        suggester._corpus = corpus
        suggester._min_chars = min_chars
        suggester._max_suggestions = max_suggestions
        suggester._retriever_specs = tuple(retriever_specs)
        suggester._retrievers = retrievers
        suggester._stored_retrievers = (
            tuple(stored_retrievers) if stored_retrievers is not None else None
        )
        suggester._weight_specs = weight_specs
        suggester._weights = weights
        suggester._artifact_provenance = artifact_provenance
        weight_specs.warn_for_retriever_names(
            [spec.name for spec in suggester._retriever_specs]
        )
        return suggester

    @classmethod
    def from_artifact(cls, artifact_dir: str | os.PathLike) -> "SAYTSuggester":
        """Load a suggester from a persisted SAYT artifact directory."""
        artifact_path = Path(artifact_dir)
        manifest = read_artifact_manifest(artifact_dir=artifact_path)
        persisted_rows = read_artifact_corpus(
            artifact_dir=artifact_path,
            corpus_file=manifest.corpus_file,
        )
        corpus = CleanCorpus.from_persisted_rows(persisted_rows)
        if corpus.size != manifest.corpus_size:
            raise ValueError("Artifact corpus size does not match manifest")

        retrievers = _load_retrievers_from_artifact(
            corpus=corpus,
            min_chars=manifest.min_chars,
            stored_retrievers=manifest.retrievers,
            artifact_dir=artifact_path,
        )

        weights = _normalised_weight_specs(manifest.weight_specs)

        artifact_provenance = SaytArtifactProvenance(
            artifact_dir=str(artifact_path),
            artifact_type=SAYT_ARTIFACT_TYPE,
            artifact_version=SAYT_ARTIFACT_VERSION,
            corpus_file=manifest.corpus_file,
            corpus_size=manifest.corpus_size,
        )
        return cls._from_state(
            corpus=corpus,
            min_chars=manifest.min_chars,
            max_suggestions=manifest.max_suggestions,
            retriever_specs=[
                stored_retriever.spec for stored_retriever in manifest.retrievers
            ],
            retrievers=retrievers,
            stored_retrievers=manifest.retrievers,
            weight_specs=manifest.weight_specs,
            weights=weights,
            artifact_provenance=artifact_provenance,
        )

    def _build_retrievers(
        self, retriever_specs: Sequence[RetrieverSpec]
    ) -> list[_ConfiguredRetriever]:
        return [
            _ConfiguredRetriever(
                name=spec.name,
                retriever=spec.build(
                    self._corpus,
                    min_chars=self._min_chars,
                ),
            )
            for spec in retriever_specs
        ]

    def _combine_suggestions(
        self,
        result_groups: Iterable[tuple[float, list[Suggestion]]],
    ) -> list[Suggestion]:
        """Combine retriever-local scores into a shared display-text score.

        Combination factors:
            - Within each retriever, scores are max-normalised by that retriever's
              top score.
            - The normalised score is multiplied by the retriever's configured
              weight.
            - For repeated display text in one retriever, the best weighted score
              is kept.
            - Across retrievers, weighted scores for the same display text are
              summed.

        Aggregation keys are display-text based, so duplicate display values are
        collapsed at this stage.
        """

        def normalise_scores(
            items: list[Suggestion], weight: float
        ) -> dict[str, float]:
            if not items:
                return {}
            max_score = max((s.score for s in items), default=0.0)
            if max_score <= 0:
                return {}
            out: dict[str, float] = {}
            for s in items:
                out[s.display_text] = max(
                    out.get(s.display_text, 0.0), s.score / max_score * weight
                )
            return out

        combined_scores: dict[str, float] = {}
        for weight, suggestions in result_groups:
            d = normalise_scores(suggestions, weight)
            for k, v in d.items():
                combined_scores[k] = combined_scores.get(k, 0.0) + v

        return [Suggestion(display_text=k, score=v) for k, v in combined_scores.items()]

    def _collect_retriever_results(
        self,
        q_norm: str,
        num_suggestions: int,
        weights: WeightSpecs | None = None,
    ) -> list[tuple[float, list[Suggestion]]]:
        result = []
        if weights is not None:
            weight_specs = _normalised_weight_specs(weights, query_length=len(q_norm))
        else:
            weight_specs = self._weights

        for configured_retriever in self._retrievers:
            start_time = time.time()

            configured_retriever_weight = weight_specs.get_weight(
                configured_retriever.name, len(q_norm)
            )

            if configured_retriever_weight == 0.0:
                continue

            result.append(
                (
                    configured_retriever_weight,
                    configured_retriever.retriever.suggest_with_scores(
                        q_norm,
                        num_suggestions=num_suggestions,
                    ),
                )
            )
            elapsed_time = time.time() - start_time
            logger.debug(
                "Retriever query time (mid level)",
                retriever_name=configured_retriever.retriever.__class__.__name__,
                query_time_ms=elapsed_time * 1000,
                num_suggestions_requested=num_suggestions,
                num_suggestions_returned=len(result[-1][1]),
                retriever_weight_for_scores=result[-1][0],
            )

        return result

    def suggest_with_scores(
        self,
        query: str | None,
        num_suggestions: int | None = None,
        weights: WeightSpecs | None = None,
    ) -> list[Suggestion]:
        """Return ranked suggestions and their combined scores.

        Args:
            query: Raw user query text.
            num_suggestions: Optional maximum number of ranked suggestions to
                return. When omitted, the configured default is used.
            weights: Optional retriever weight override. When supplied,
                the suggester will reweight the configured retrievers for this
                query only. The retriever order is preserved, but the weights are
                normalised to sum to 1.0.

        Returns:
            A list of combined suggestions ordered by descending score. Returns
            an empty list when the normalised query is shorter than
            ``min_chars``.

        Notes:
            - Each retriever is asked for ``num_suggestions * 5`` candidates to
              improve cross-retriever score pairing before final truncation.
            - Final ranking and cutoff tie handling are delegated to
              ``take_with_ties`` using corpus display-text duplication counts.
            - Output is display-text deduplicated by ``take_with_ties``.
        """
        if num_suggestions is None:
            num_suggestions = self._max_suggestions
        q_norm = _normalise(query)
        if len(q_norm) < self._min_chars:
            return []

        results_by_kind = self._collect_retriever_results(
            q_norm,
            num_suggestions=num_suggestions * 5,
            # collect more to allow pairing up scores with other retrievers
            weights=weights,
        )

        combined_result = self._combine_suggestions(results_by_kind)
        return take_with_ties(
            combined_result, num_suggestions, self._corpus.display_text_value_counts
        )

    def suggest(
        self,
        query: str | None,
        num_suggestions: int | None = None,
        weights: WeightSpecs | None = None,
    ) -> list[str]:
        """Return display-text-deduplicated suggestions.

        Args:
            query: Raw user query text.
            num_suggestions: Optional maximum number of display values to
                return. When omitted, the configured default is used.
            weights: Optional retriever weight override. When supplied,
                the suggester will reweight the configured retrievers for this
                query only. The retriever order is preserved, but the weights are
                normalised to sum to 1.0. This is useful for testing or for
                temporarily adjusting the relative influence of retrievers without
                rebuilding the suggester.

        Returns:
            A list of display-text suggestions ordered by descending combined
            score, while preserving ties at the cutoff.
        """
        start_time = time.time()

        if num_suggestions is None:
            num_suggestions = self._max_suggestions
        results = self.suggest_with_scores(
            query,
            num_suggestions=num_suggestions,
            weights=weights,
        )
        elapsed_time = time.time() - start_time
        logger.debug(
            "Suggest query time (top level)",
            query_time_ms=elapsed_time * 1000,
            num_suggestions_requested=num_suggestions,
            num_suggestions_returned=len(results),
        )

        return [s.display_text for s in results]

    def get_config(self) -> SaytConfiguration:
        """Return a rich runtime summary of this suggester.

        Returns:
            A summary of global settings, corpus details, retriever
            configuration, and any artifact provenance available for this
            suggester.
        """
        stored_retrievers: Sequence[StoredRetrieverSpec | None]
        if self._stored_retrievers is None:
            stored_retrievers = [None] * len(self._retriever_specs)
        else:
            stored_retrievers = list(self._stored_retrievers)

        retrievers = [
            _build_retriever_summary(
                spec=spec,
                configured_retriever=configured_retriever,
                stored_retriever=stored_retriever,
            )
            for spec, configured_retriever, stored_retriever in zip(
                self._retriever_specs,
                self._retrievers,
                stored_retrievers,
                strict=True,
            )
        ]

        return SaytConfiguration(
            settings=SaytGlobalSettings(
                min_chars=self._min_chars,
                max_suggestions=self._max_suggestions,
            ),
            corpus=SaytCorpusSummary(
                size=self._corpus.size,
                unique_display_texts=len(self._corpus.display_text_value_counts),
                max_duplication=max(
                    self._corpus.display_text_value_counts.values(), default=0
                ),
            ),
            retrievers=retrievers,
            artifact_provenance=(
                self._artifact_provenance.model_copy(deep=True)
                if self._artifact_provenance is not None
                else None
            ),
        )

    def update_weights(
        self,
        weights: WeightSpecs,
    ) -> None:
        """Update the retriever weights for this suggester.

        Args:
            weights: The new retriever weights to apply to this suggester.

        Raises:
            ValueError: If the provided weights are invalid or empty.
        """
        weights.warn_for_retriever_names([spec.name for spec in self._retriever_specs])
        self._weight_specs = weights
        self._weights = _normalised_weight_specs(self._weight_specs)


def _normalised_weight_specs(
    weight_specs: WeightSpecs,
    query_length: int | None = None,
) -> WeightSpecs:
    if not weight_specs.specs:
        raise ValueError("At least one retriever weight must be configured")

    weights_dict = weight_specs.get_normalised_weights(query_length=query_length)
    updated_specs = []
    for spec in weight_specs.specs:
        if spec.retriever_name in weights_dict:
            spec_type = type(spec)
            updated_specs.append(spec_type(weights=weights_dict[spec.retriever_name]))

    return WeightSpecs(specs=updated_specs)


def _load_retrievers_from_artifact(
    *,
    corpus: CleanCorpus,
    min_chars: int,
    stored_retrievers: Sequence[StoredRetrieverSpec],
    artifact_dir: Path,
) -> list[_ConfiguredRetriever]:
    """Restore runtime retrievers from a persisted SAYT artifact."""
    return [
        _ConfiguredRetriever(
            name=stored_retriever.spec.name,
            retriever=load_retriever_from_artifact(
                corpus=corpus,
                min_chars=min_chars,
                stored_retriever=stored_retriever,
                artifact_dir=artifact_dir,
            ),
        )
        for stored_retriever in stored_retrievers
    ]


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable_value(item) for item in value]
    return str(value)


def _summarise_retriever_config(spec: RetrieverSpec) -> dict[str, Any]:
    if is_dataclass(spec):
        items = (
            (field.name, getattr(spec, field.name))
            for field in fields(spec)
            if field.name not in {"name"}
        )
        return {key: _jsonable_value(value) for key, value in items}

    raw_config = getattr(spec, "__dict__", None)
    if isinstance(raw_config, dict):
        return {
            str(key): _jsonable_value(value)
            for key, value in raw_config.items()
            if key not in {"name"}
        }
    return {}


def _build_retriever_summary(
    *,
    spec: RetrieverSpec,
    configured_retriever: _ConfiguredRetriever,
    stored_retriever: StoredRetrieverSpec | None,
) -> SaytRetrieverSummary:
    artifact_provenance = None
    config = _summarise_retriever_config(spec)
    if stored_retriever is not None:
        artifact_provenance = SaytRetrieverArtifactProvenance(
            artifact_type=stored_retriever.spec.name,
            path=stored_retriever.path,
            config=_summarise_retriever_config(stored_retriever.spec),
        )

    return SaytRetrieverSummary(
        name=spec.name,
        spec_type=type(spec).__name__,
        retriever_type=type(configured_retriever.retriever).__name__,
        config=config,
        artifact_provenance=artifact_provenance,
    )
