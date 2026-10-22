"""Tests for SAYT retrieval and ranking behaviour."""

# ruff: noqa: PLR2004

# pylint: disable=protected-access,redefined-outer-name,too-few-public-methods,C0116,W0613

import csv
import shutil
from pathlib import Path

import numpy as np
import pytest
from classifai.vectorisers import VectoriserBase

import survey_assist_embed_core.adapters.classifai.vectoriser as vectoriser_module
from survey_assist_embed_core.sayt import NgramRetrieverSpec, PrefixRetrieverSpec
from survey_assist_embed_core.sayt.core import CleanCorpus
from survey_assist_embed_core.sayt.indexes import (
    DenseVectorIndex,
    _CharNgramVectoriser,
    _derive_num_retrieved_based_on_duplication,
    load_semantic_index,
)
from survey_assist_embed_core.sayt.retrievers import (
    NgramRetriever,
    PrefixRetriever,
    SemanticRetriever,
    _lookup_prefix,
    _PrefixTrieNode,
)
from survey_assist_embed_core.sayt.suggester import SAYTSuggester
from survey_assist_embed_core.sayt.weight_specs import (
    NgramWeightSpec,
    PrefixWeightSpec,
    WeightSpecs,
)


def test_prefix_full_string_match_ranks_expected_terms(small_corpus):
    """Return relevant prefix matches and exclude unrelated terms."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[PrefixRetrieverSpec()],
        weights=WeightSpecs(specs=[PrefixWeightSpec()]),
    )
    results = suggester.suggest("car")

    assert "Car Wash" in results
    assert "Car Waxing" in results
    assert "Dog grooming" not in results


def test_duplicate_terms_increase_rank_via_counts(small_corpus):
    """Prefer duplicated underlying search terms when scores tie."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[PrefixRetrieverSpec()],
        weights=WeightSpecs(specs=[PrefixWeightSpec()]),
    )
    results = suggester.suggest("car w")
    assert results[0] == "Car Waxing"


def test_duplicate_display_variants_are_returned_shorter_first(small_corpus):
    """Rank duplicate display variants using the configured tie-breakers."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[PrefixRetrieverSpec()],
        weights=WeightSpecs(specs=[PrefixWeightSpec()]),
    )
    results = suggester.suggest("car wa")
    index_1 = results.index("Car Wash")
    index_2 = results.index("CAR WASH (duplicate)")
    assert index_1 < index_2


def test_fuzzy_prefix_can_recover_from_simple_typo(small_corpus):
    """Recover expected results when the query has a small typo."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[PrefixRetrieverSpec()],
        weights=WeightSpecs(specs=[PrefixWeightSpec()]),
    )
    results = suggester.suggest("carpentey")
    assert "Carpentry services" in results


def test_ngram_recovers_from_typo_when_prefix_does_not_match(small_corpus):
    """Use n-gram retrieval when typoed input misses prefix matching."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[PrefixRetrieverSpec(), NgramRetrieverSpec(n=3, max_df=1.0)],
        max_suggestions=5,
        weights=WeightSpecs(specs=[PrefixWeightSpec(), NgramWeightSpec()]),
    )
    results = suggester.suggest("groming")
    assert results[0] == "Dog grooming"


class _StubVectoriser(VectoriserBase):
    def __init__(self, output):
        self._output = output

    def transform(self, texts):
        return self._output


class _StubSearchResults:
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, column):
        return np.array([row[column] for row in self._rows])

    def to_dict(self, orient="dict"):
        assert orient == "records"
        return self._rows


class _StubVectorStore:
    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def search(self, query, n_results=10):
        self.calls.append((query, n_results))
        return _StubSearchResults(self._rows)


def test_prefix_retriever_returns_empty_for_short_queries(small_corpus):
    """Skip prefix work when the query is shorter than the minimum."""
    corpus = CleanCorpus.model_validate(small_corpus)

    assert not PrefixRetriever(corpus, min_chars=4).suggest_with_scores(
        "car", num_suggestions=5
    )


def test_prefix_retriever_handles_empty_prefix_candidates(small_corpus):
    """Return no prefix suggestions when the query is empty."""
    corpus = CleanCorpus.model_validate(small_corpus)
    retriever = PrefixRetriever(corpus, min_chars=0)

    results = retriever.suggest_with_scores("", num_suggestions=5)

    assert results == []


def test_lookup_prefix_returns_empty_when_prefix_path_ends_missing() -> None:
    """Return no matches when the final trie step resolves to no child node."""
    assert _lookup_prefix(_PrefixTrieNode(), "z") == set()


def test_dense_candidate_derivation_returns_zero_for_invalid_inputs() -> None:
    """Return zero candidates when no suggestions or corpus rows are available."""
    assert _derive_num_retrieved_based_on_duplication(0, 3, 10) == 0
    assert _derive_num_retrieved_based_on_duplication(5, 3, 0) == 0


def test_prefix_retriever_keeps_ties_at_cutoff():
    """Return all prefix candidates tied on score at the suggestion cutoff."""
    corpus = CleanCorpus.model_validate(
        [("car wash", "Car Wash"), ("car waxing", "Car Waxing")]
    )

    results = PrefixRetriever(corpus, min_chars=3).suggest_with_scores(
        "car", num_suggestions=1
    )

    assert [s.display_text for s in results] == ["Car Wash", "Car Waxing"]


def test_onnx_vectoriser_handles_one_dimensional_output(monkeypatch):
    """Normalise a single returned embedding into a 2D unit vector."""

    class _StubTextEmbedding:
        def __init__(self, model_name, **kwargs):
            _ = model_name

        def embed(self, texts):
            _ = texts
            return [np.array([3.0, 4.0], dtype=np.float32)]

    vectoriser_module._ONNX_MODEL_CACHE.clear()
    monkeypatch.setattr(vectoriser_module, "TextEmbedding", _StubTextEmbedding)

    vectoriser = vectoriser_module.OnnxVectoriser(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    vectors = vectoriser.transform("query")

    assert vectors.shape == (1, 2)
    assert vectors[0] == pytest.approx(np.array([0.6, 0.8]))


def test_normalise_vectors_preserves_two_dimensional_output():
    """Normalise batched vectors without reshaping an existing matrix."""
    vectors = vectoriser_module.normalise_vectors(np.array([[3.0, 4.0], [5.0, 12.0]]))

    assert vectors.shape == (2, 2)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx(np.array([1.0, 1.0]))


def test_onnx_vectoriser_reuses_cached_text_embedding(monkeypatch):
    """Reuse a single TextEmbedding instance for the same resolved model name."""
    captured = []

    class _StubTextEmbedding:
        def __init__(self, model_name, **kwargs):
            captured.append(model_name)

        def embed(self, texts):
            _ = texts
            return [np.array([1.0, 0.0], dtype=np.float32)]

    vectoriser_module._ONNX_MODEL_CACHE.clear()
    monkeypatch.setattr(vectoriser_module, "TextEmbedding", _StubTextEmbedding)

    first = vectoriser_module.OnnxVectoriser("sentence-transformers/all-MiniLM-L6-v2")
    second = vectoriser_module.OnnxVectoriser("sentence-transformers/all-MiniLM-L6-v2")

    assert captured == ["sentence-transformers/all-MiniLM-L6-v2"]
    assert first.model is second.model


def test_char_ngram_vectoriser_accepts_single_string_input():
    """Transform a scalar string query into a single-row dense matrix."""
    vectoriser = _CharNgramVectoriser(["car wash", "dog grooming"], n=3, max_df=1.0)

    vectors = vectoriser.transform("car")

    assert vectors.shape[0] == 1


def test_ngram_retriever_returns_empty_for_empty_query_vector(
    monkeypatch, small_corpus
):
    """Return no suggestions when the vector store yields no matches."""
    corpus = CleanCorpus.model_validate(small_corpus)
    retriever = NgramRetriever.__new__(NgramRetriever)
    retriever._corpus = corpus
    retriever._min_chars = 3
    retriever._index = DenseVectorIndex(
        _vector_store=_StubVectorStore([]),
        _num_vectors=1,
        _corpus=corpus,
    )

    assert not retriever.suggest_with_scores("car", num_suggestions=3)


def test_ngram_retriever_returns_empty_for_short_queries():
    """Stop before vectorisation when the query is shorter than min_chars."""
    retriever = NgramRetriever.__new__(NgramRetriever)
    retriever._min_chars = 4

    assert not retriever.suggest_with_scores("car", num_suggestions=3)


def test_ngram_retriever_returns_empty_for_empty_similarity_matrix():
    """Stop early when the dense index contains no stored vectors."""
    retriever = NgramRetriever.__new__(NgramRetriever)
    retriever._corpus = CleanCorpus.model_validate([("car wash", "Car Wash")])
    retriever._min_chars = 3
    retriever._index = DenseVectorIndex(
        _vector_store=_StubVectorStore([]),
        _num_vectors=0,
        _corpus=retriever._corpus,
    )

    assert not retriever.suggest_with_scores("car", num_suggestions=3)


def test_dense_retriever_keeps_ties_at_cutoff(small_corpus):
    """Return all dense candidates tied on score at the suggestion cutoff."""
    corpus = CleanCorpus.model_validate(small_corpus)
    retriever = NgramRetriever.__new__(NgramRetriever)
    retriever._corpus = corpus
    retriever._min_chars = 3
    retriever._index = DenseVectorIndex(
        _vector_store=_StubVectorStore(
            [
                {"doc_label": corpus.rows[0][1], "score": 0.9},
                {"doc_label": corpus.rows[1][1], "score": 0.9},
                {"doc_label": corpus.rows[2][1], "score": 0.2},
            ]
        ),
        _num_vectors=3,
        _corpus=corpus,
    )

    results = retriever.suggest_with_scores("car", num_suggestions=1)

    assert {s.display_text for s in results} == {corpus.rows[0][1], corpus.rows[1][1]}


def test_dense_vector_index_builds_persistent_filespace(
    monkeypatch, tmp_path, small_corpus
):
    """Persist dense indexes even when the output folder is replaced first."""
    captured = {}
    corpus = CleanCorpus.model_validate(small_corpus)
    output_dir = tmp_path / "ngram"
    output_dir.mkdir()

    class _StubPersistentVectorStore:
        # pylint: disable=too-many-arguments
        def __init__(  # noqa: PLR0913
            self,
            *,
            file_name,
            data_type,
            vectoriser,
            batch_size,
            output_dir,
            overwrite,
            skip_save,
            hooks,
            quiet_mode,
        ):
            captured["file_name"] = file_name
            captured["data_type"] = data_type
            captured["vectoriser_type"] = type(vectoriser).__name__
            captured["batch_size"] = batch_size
            captured["output_dir"] = output_dir
            captured["overwrite"] = overwrite
            captured["skip_save"] = skip_save
            captured["hooks"] = hooks
            captured["quiet_mode"] = quiet_mode
            output_path = Path(output_dir)
            if output_path.is_dir() and overwrite:
                shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(file_name, encoding="utf-8") as input_file:
                captured["rows"] = list(csv.DictReader(input_file))
            (output_path / "metadata.json").write_text("{}", encoding="utf-8")
            (output_path / "vectors.parquet").write_text("dummy", encoding="utf-8")
            self.num_vectors = len(captured["rows"])

    monkeypatch.setattr(
        "survey_assist_embed_core.sayt.indexes.VectorStore",
        _StubPersistentVectorStore,
    )

    index = DenseVectorIndex.from_corpus(
        corpus=corpus,
        vectoriser=_StubVectoriser(np.array([[1.0, 0.0]])),
        output_dir=output_dir,
        overwrite=True,
    )

    assert index._num_vectors == len(corpus.rows)
    assert index._max_duplication == 2  # "Car Waxing" appears twice in small_corpus
    assert Path(captured["file_name"]).name == "corpus.csv"
    assert Path(captured["file_name"]).parent != output_dir
    assert captured["data_type"] == "csv"
    assert captured["vectoriser_type"] == "_StubVectoriser"
    assert captured["batch_size"] == 128
    assert captured["output_dir"] == str(output_dir)
    assert captured["overwrite"] is True
    assert captured["skip_save"] is False
    assert captured["hooks"] is None
    assert captured["quiet_mode"] is True
    assert captured["rows"] == [
        {"label": display_text, "text": search_text}
        for search_text, display_text in corpus.rows
    ]


def test_dense_vector_index_query_scales_n_results_by_max_duplication(small_corpus):
    """Fetch num_suggestions * _max_duplication candidates from the vector store."""
    corpus = CleanCorpus.model_validate(small_corpus)
    stub_store = _StubVectorStore([])
    index = DenseVectorIndex(
        _vector_store=stub_store,
        _num_vectors=100,
        _corpus=corpus,
        _max_duplication=3,
    )

    index.query("car", num_suggestions=5)

    assert stub_store.calls[0][1] == 10  # 5 * 2


def test_dense_vector_index_loads_existing_filespace(
    monkeypatch, tmp_path, small_corpus
):
    """Load a persisted dense index via ClassifAI's filespace API."""
    captured = {}
    corpus = CleanCorpus.model_validate(small_corpus)
    folder_path = tmp_path / "existing-ngram"

    class _StubLoadedVectorStore:
        num_vectors = 7

    def _fake_from_filespace(*, folder_path, vectoriser, hooks, quiet_mode):
        captured["folder_path"] = folder_path
        captured["vectoriser_type"] = type(vectoriser).__name__
        captured["hooks"] = hooks
        captured["quiet_mode"] = quiet_mode
        return _StubLoadedVectorStore()

    monkeypatch.setattr(
        "survey_assist_embed_core.sayt.indexes.VectorStore.from_filespace",
        _fake_from_filespace,
    )

    index = DenseVectorIndex.from_filespace(
        corpus=corpus,
        folder_path=folder_path,
        vectoriser=_StubVectoriser(np.array([[1.0, 0.0]])),
    )

    assert index._num_vectors == 7
    assert index._max_duplication == 2  # "Car Waxing" appears twice in small_corpus
    assert index._corpus is corpus
    assert captured == {
        "folder_path": str(folder_path),
        "vectoriser_type": "_StubVectoriser",
        "hooks": None,
        "quiet_mode": True,
    }


def test_semantic_retriever_builds_index_with_wrapped_vectoriser(
    monkeypatch, small_corpus
):
    """Use the vectoriser factory before building the dense index."""
    captured = {}
    corpus = CleanCorpus.model_validate(small_corpus)

    class _StubSemanticVectoriser:
        def transform(self, texts):
            _ = texts
            return np.array([[1.0, 0.0]])

    def _fake_build_vectoriser(*, embedding_model_name, vectoriser_class=None):
        captured["model_name"] = embedding_model_name
        captured["vectoriser_class"] = getattr(
            vectoriser_class,
            "value",
            vectoriser_class,
        )
        return _StubSemanticVectoriser()

    def _fake_build_dense_vector_index(
        *,
        corpus,
        vectoriser,
        output_dir=None,
        overwrite=True,
    ):
        captured["vectoriser_type"] = type(vectoriser).__name__
        captured["output_dir"] = output_dir
        captured["overwrite"] = overwrite
        return DenseVectorIndex(
            _vector_store=_StubVectorStore([]),
            _num_vectors=1,
            _corpus=corpus,
        )

    monkeypatch.setattr(
        "survey_assist_embed_core.sayt.indexes.build_vectoriser",
        _fake_build_vectoriser,
    )
    monkeypatch.setattr(
        "survey_assist_embed_core.sayt.indexes.DenseVectorIndex.from_corpus",
        _fake_build_dense_vector_index,
    )

    retriever = SemanticRetriever(
        corpus,
        model="all-MiniLM-L6-v2",
        vectoriser_class="HF",
        min_chars=3,
    )

    assert captured == {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "vectoriser_class": "HF",
        "vectoriser_type": "_StubSemanticVectoriser",
        "output_dir": None,
        "overwrite": True,
    }
    assert retriever._min_chars == 3


def test_load_semantic_index_loads_existing_filespace_with_wrapped_vectoriser(
    monkeypatch, tmp_path, small_corpus
):
    """Use the vectoriser factory before loading a persisted semantic index."""
    captured = {}
    corpus = CleanCorpus.model_validate(small_corpus)
    folder_path = tmp_path / "existing-semantic"

    class _StubSemanticVectoriser:
        def transform(self, texts):
            _ = texts
            return np.array([[1.0, 0.0]])

    def _fake_build_vectoriser(*, embedding_model_name, vectoriser_class=None):
        captured["model_name"] = embedding_model_name
        captured["vectoriser_class"] = getattr(
            vectoriser_class,
            "value",
            vectoriser_class,
        )
        return _StubSemanticVectoriser()

    def _fake_load_dense_vector_index(*, corpus, folder_path, vectoriser):
        captured["corpus"] = corpus
        captured["folder_path"] = folder_path
        captured["vectoriser_type"] = type(vectoriser).__name__
        return DenseVectorIndex(
            _vector_store=_StubVectorStore([]),
            _num_vectors=2,
            _corpus=corpus,
        )

    monkeypatch.setattr(
        "survey_assist_embed_core.sayt.indexes.build_vectoriser",
        _fake_build_vectoriser,
    )
    monkeypatch.setattr(
        "survey_assist_embed_core.sayt.indexes.DenseVectorIndex.from_filespace",
        _fake_load_dense_vector_index,
    )

    index = load_semantic_index(
        corpus,
        model="all-MiniLM-L6-v2",
        vectoriser_class="HF",
        folder_path=folder_path,
    )

    assert index._num_vectors == 2
    assert captured == {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "vectoriser_class": "HF",
        "corpus": corpus,
        "folder_path": folder_path,
        "vectoriser_type": "_StubSemanticVectoriser",
    }


def test_semantic_retriever_returns_empty_for_short_queries():
    """Stop before vectorisation when the semantic query is too short."""
    retriever = SemanticRetriever.__new__(SemanticRetriever)
    retriever._min_chars = 4

    assert not retriever.suggest_with_scores("car", num_suggestions=3)


def test_semantic_retriever_returns_empty_for_empty_query_vector(small_corpus):
    """Return no suggestions when semantic vector search yields no matches."""
    corpus = CleanCorpus.model_validate(small_corpus)
    retriever = SemanticRetriever.__new__(SemanticRetriever)
    retriever._corpus = corpus
    retriever._min_chars = 3
    retriever._index = DenseVectorIndex(
        _vector_store=_StubVectorStore([]),
        _num_vectors=1,
        _corpus=corpus,
    )

    assert not retriever.suggest_with_scores("car", num_suggestions=3)


def test_semantic_retriever_returns_empty_for_empty_similarity_matrix():
    """Stop early when semantic retrieval has no stored vectors."""
    retriever = SemanticRetriever.__new__(SemanticRetriever)
    retriever._corpus = CleanCorpus.model_validate([("car wash", "Car Wash")])
    retriever._min_chars = 3
    retriever._index = DenseVectorIndex(
        _vector_store=_StubVectorStore([]),
        _num_vectors=0,
        _corpus=retriever._corpus,
    )

    assert not retriever.suggest_with_scores("car", num_suggestions=3)
