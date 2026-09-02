from langchain_core.documents import Document

from app.rag.reranker import rerank_hits
from app.rag.vectorstore import SearchHit


def _hit(text: str, *, score: float = 0.5) -> SearchHit:
    return SearchHit(
        document=Document(page_content=text, metadata={"source": "a.md"}),
        score=score,
        vector_sim=score,
        bm25_score=0.0,
    )


def test_rerank_disabled_path_returns_truncated_hits(monkeypatch) -> None:
    hits = [_hit("alpha"), _hit("beta"), _hit("gamma")]
    ranked = rerank_hits(
        "query",
        hits,
        top_k=2,
        model_name="unused",
        timeout_seconds=1.0,
    )
    assert len(ranked) == 2
    assert ranked[0].document.page_content == "alpha"


def test_rerank_reorders_by_mock_scores(monkeypatch) -> None:
    hits = [_hit("wrong course"), _hit("right course")]

    def fake_predict(model_name: str, pairs: list[tuple[str, str]]) -> list[float]:
        return [0.1, 0.9]

    class _Immediate:
        def __init__(self, value):
            self._value = value

        def result(self, timeout=None):
            return self._value

    monkeypatch.setattr("app.rag.reranker._predict_scores", fake_predict)
    monkeypatch.setattr(
        "app.rag.reranker._executor.submit",
        lambda fn, *args, **kwargs: _Immediate(fn(*args)),
    )
    ranked = rerank_hits(
        "right course",
        hits,
        top_k=2,
        model_name="mock",
        timeout_seconds=1.0,
    )
    assert ranked[0].document.page_content == "right course"
    assert ranked[0].score == 0.9


def test_rerank_falls_back_on_failure(monkeypatch) -> None:
    hits = [_hit("first"), _hit("second"), _hit("third")]

    def boom(*_args, **_kwargs):
        raise RuntimeError("model down")

    monkeypatch.setattr("app.rag.reranker._predict_scores", boom)
    ranked = rerank_hits(
        "query",
        hits,
        top_k=2,
        model_name="mock",
        timeout_seconds=1.0,
    )
    assert [hit.document.page_content for hit in ranked] == ["first", "second"]


def test_rerank_preserves_low_confidence(monkeypatch) -> None:
    hits = [
        SearchHit(
            document=_hit("wrong course").document,
            score=0.4,
            vector_sim=0.4,
            bm25_score=0.0,
            low_confidence=True,
        ),
        _hit("right course"),
    ]

    def fake_predict(model_name: str, pairs: list[tuple[str, str]]) -> list[float]:
        return [0.2, 0.9]

    class _Immediate:
        def __init__(self, value):
            self._value = value

        def result(self, timeout=None):
            return self._value

    monkeypatch.setattr("app.rag.reranker._predict_scores", fake_predict)
    monkeypatch.setattr(
        "app.rag.reranker._executor.submit",
        lambda fn, *args, **kwargs: _Immediate(fn(*args)),
    )
    ranked = rerank_hits(
        "right course",
        hits,
        top_k=2,
        model_name="mock",
        timeout_seconds=1.0,
    )
    assert ranked[0].document.page_content == "right course"
    assert any(hit.low_confidence for hit in ranked)


def test_warmup_reranker_loads_once(monkeypatch) -> None:
    loads: list[str] = []

    def fake_load(model_name: str):
        loads.append(model_name)
        return object()

    monkeypatch.setattr("app.rag.reranker._load_cross_encoder", fake_load)
    from app.rag.reranker import warmup_reranker

    warmup_reranker("mock-model")
    warmup_reranker("mock-model")
    assert loads == ["mock-model"]
