"""Fusion, vector search, the eval's metrics and its embedding cache (invented data only:
never a real question, answer or expected id)."""

from pathlib import Path

import pytest

from corpusqa.config import RRF_BM25_WEIGHT, RRF_DENSE_WEIGHT, RRF_K
from corpusqa.ingest import chunking as chunking_module
from corpusqa.ingest.chunking import ChunkSpec
from corpusqa.llm.fakes import FakeEmbedder
from corpusqa.obs.testing import RecordingTracer
from corpusqa.retrieval.hybrid import Rankings, rrf, to_documents
from corpusqa.store.sqlite import SqliteVectorCache
from corpusqa.store.vectors import VectorIndex, from_blob, to_blob
from eval.retrieval import (
    CachedEmbedder,
    Fusion,
    Variant,
    headline,
    history_metrics,
    parse_args,
    render_markdown,
)
from eval.retrieval_metrics import QuestionRun, gold_ranks, score, score_all


# ---------------------------------------------------------------- fusion
def test_rrf_sums_reciprocal_ranks() -> None:
    fused = rrf([[1, 2, 3], [3, 1]], k=RRF_K)
    # 1: 1/61 + 1/62; 3: 1/63 + 1/61; 2: 1/62
    assert fused == [1, 3, 2]


def test_rrf_ties_go_to_the_lower_id_and_one_list_keeps_its_order() -> None:
    assert rrf([[5], [4]]) == [4, 5]
    assert rrf([[9, 7, 8], []]) == [9, 7, 8]


def test_rrf_weights_change_the_order() -> None:
    assert rrf([[1], [2]]) == [1, 2]  # a tie: the lower id first
    assert rrf([[1], [2]], weights=[1.0, 2.0]) == [2, 1]  # 2/61 beats 1/61
    assert rrf([[1], [2]], weights=[1.0, 1.0]) == rrf([[1], [2]])  # 1.0 each = plain RRF
    # plain: 6 = 1/62 + 1/61, 5 = 1/61, 1 = 1/62
    assert rrf([[5, 6], [6, 1]]) == [6, 5, 1]
    # second list x3: 6 = 1/62 + 3/61, 1 = 3/62, 5 = 1/61
    assert rrf([[5, 6], [6, 1]], weights=[1.0, 3.0]) == [6, 1, 5]


def test_rrf_an_id_only_in_a_zero_weight_list_sorts_after_every_scored_id() -> None:
    # 1 is the lowest id, but only the weight-0 list holds it
    assert rrf([[5, 6], [6, 1]], weights=[1.0, 0.0]) == [5, 6, 1]
    assert rrf([[5, 6], [6, 1]], weights=[0.0, 1.0]) == [6, 1, 5]


def test_rrf_needs_one_weight_per_list() -> None:
    with pytest.raises(ValueError, match="weights"):
        rrf([[1], [2]], weights=[1.0])
    with pytest.raises(ValueError, match="weights"):
        rrf([[1]], weights=[1.0, 1.0])


def test_chunks_map_to_documents_at_their_best_rank() -> None:
    chunk_doc = {1: "a", 2: "b", 3: "a", 4: "c"}
    assert to_documents([3, 2, 1, 4], chunk_doc) == ("a", "b", "c")


# ---------------------------------------------------------------- vectors
def test_vector_index_returns_the_nearest_rows_best_first() -> None:
    rows = [(10, to_blob([1.0, 0.0])), (11, to_blob([0.0, 1.0])), (12, to_blob([0.6, 0.8]))]
    index = VectorIndex.from_rows(rows, dims=2)
    hits = index.search([0.0, 2.0], k=2)  # the query is normalised first
    assert [c for c, _ in hits] == [11, 12]
    assert hits[0][1] == pytest.approx(1.0) and hits[1][1] == pytest.approx(0.8)
    assert len(index.search([1.0, 0.0], k=10)) == 3
    assert index.search([0.0, 0.0], k=2) == []
    with pytest.raises(ValueError, match="values"):
        index.search([1.0, 0.0, 0.0], k=1)
    assert from_blob(to_blob([0.25, -0.5])) == [0.25, -0.5]


# ---------------------------------------------------------------- metrics
def run(qid: str, gold: set[str], ranked: list[str]) -> QuestionRun:
    docs = tuple(ranked)
    return QuestionRun(qid, frozenset(gold), Rankings(bm25=docs, dense=(), hybrid=docs))


def test_gold_ranks_lists_ranks_best_first_then_misses() -> None:
    assert gold_ranks(["x", "g2", "y", "g1"], frozenset({"g1", "g2", "g3"})) == [2, 4, None]


def test_recall_pools_all_gold_documents_and_all_gold_counts_questions() -> None:
    filler = [f"f{i}" for i in range(20)]
    runs = [
        run("q1", {"a"}, ["a", *filler]),  # gold at rank 1
        run("q2", {"b", "c"}, [*filler[:6], "b", *filler[6:], "c"]),  # ranks 7 and 21
        run("q3", {"d"}, filler),  # never found
        run("q4", set(), filler),  # no gold: left out
    ]
    s = score(runs, "bm25", ks=(5, 10, 20), all_gold_k=10)
    assert s.questions == 3 and s.gold_documents == 4
    assert s.recall == {5: 1 / 4, 10: 2 / 4, 20: 2 / 4}
    assert s.all_gold == pytest.approx(1 / 3)
    assert s.mrr == pytest.approx((1 + 1 / 7 + 0) / 3)
    # per question: q1 1/1 at every k; q2 0/2 at 5, 1/2 at 10 and 20; q3 0/1
    assert s.q_recall == pytest.approx({5: 1 / 3, 10: 1.5 / 3, 20: 1.5 / 3})
    empty = score_all(runs, ks=(10,), all_gold_k=10)["dense"]
    assert empty.recall == {10: 0.0} and empty.q_recall == {10: 0.0} and empty.mrr == 0.0


def test_per_question_recall_weighs_every_question_the_same() -> None:
    filler = [f"f{i}" for i in range(10)]
    many = {f"m{i}" for i in range(8)}
    runs = [run("q0", many, sorted(many))]  # 8 gold documents, all in the top 10
    runs += [run(f"q{i}", {f"g{i}"}, filler) for i in range(1, 8)]  # 1 gold each, never found
    s = score(runs, "hybrid", ks=(10,), all_gold_k=10)
    assert s.questions == 8 and s.gold_documents == 15
    assert s.recall[10] == pytest.approx(8 / 15)  # pooled: q0's 8 documents dominate
    assert s.q_recall[10] == pytest.approx(1 / 8)  # per question: one question of eight


# ---------------------------------------------------------------- embedding cache
def test_cached_embedder_embeds_each_text_once_and_keys_by_role(tmp_path: Path) -> None:
    inner = FakeEmbedder(dims=8)
    cached = CachedEmbedder(inner, SqliteVectorCache(tmp_path / "c.sqlite"), "1", workers=2,
                            batch_size=2)  # fmt: skip
    texts = ["alpha one", "beta two", "gamma three", "alpha one"]
    first = cached.embed(texts, "document")
    assert first[0] == pytest.approx(first[3])
    assert sum(len(batch) for _, batch in inner.calls) == 3  # the repeat is sent once
    assert cached.counts["misses"] == 3 and cached.counts["requests"] == 2
    again = cached.embed(texts, "document")
    assert again == first and len(inner.calls) == 2  # all from the cache
    cached.embed(["alpha one"], "query")  # another role is another key
    assert inner.calls[-1] == ("query", ("alpha one",))
    assert cached.key("t", "query") != cached.key("t", "document")
    assert first[1] == pytest.approx(inner.embed(["beta two"], "document")[0], abs=1e-6)


def test_cached_embedder_traces_generation_calls_under_the_current_span(tmp_path: Path) -> None:
    class TracedFake(FakeEmbedder):
        def embed(self, texts, role):  # type: ignore[no-untyped-def]
            from corpusqa import obs

            with obs.generation("embed", model=self.model, input={"texts": list(texts)}):
                return super().embed(texts, role)

    tracer = RecordingTracer()
    cached = CachedEmbedder(TracedFake(dims=4), SqliteVectorCache(tmp_path / "c.sqlite"), "1",
                            workers=3, batch_size=1)  # fmt: skip
    with tracer.trace("eval_retrieval"), tracer.span("embed_phase") as span:
        cached.blobs(["a", "b", "c"], "document")
    [parent] = tracer.named("embed_phase")
    generations = [o for o in tracer.children(parent) if o.kind == "generation"]
    assert len(generations) == 3 and span.trace_id == parent.trace_id


# ---------------------------------------------------------------- variant names and report
def test_variant_names_say_what_differs_from_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chunking_module, "CHUNK_HEADER_FIELDS", False)  # the names below
    assert Variant(ChunkSpec("recursive"), True, False).name == "recursive-512"
    assert Variant(ChunkSpec("fixed", 512), True, False).name == "fixed-512-o51"
    assert Variant(ChunkSpec("whole"), True, False).name == "whole-6000"
    off = Variant(ChunkSpec("recursive", header=False), promote=False, screen=False)
    assert off.name == "recursive-512-noheader-nopromote"
    fields = Variant(ChunkSpec("recursive", fields=True), True, False)
    assert fields.name == "recursive-512-fields" and fields.describe()["header_fields"] is True
    assert Variant(ChunkSpec("recursive"), True, False).describe()["header_fields"] is False
    # With the switch on, a spec that does not say follows it, and the name shows it.
    monkeypatch.setattr(chunking_module, "CHUNK_HEADER_FIELDS", True)
    assert Variant(ChunkSpec("recursive"), True, False).name == "recursive-512-fields"
    assert Variant(ChunkSpec("recursive", fields=False), True, False).name == "recursive-512"


def test_report_and_history_line_render_without_dense() -> None:
    result = {
        "variant": "v", "set": "tune", "questions_with_gold": 1, "gold_documents": 1,
        "questions_without_gold_skipped": 0, "git_commit": "abc", "timestamp": "t",
        "settings": {}, "retriever": {"top_chunks": 50, "rrf_k": 60}, "dense": False,
        "scores": {"bm25": {"recall@5": 1.0, "recall@10": 1.0, "recall@20": 1.0,
                            "q_recall@5": 0.5, "q_recall@10": 0.5, "q_recall@20": 0.5,
                            "all_gold@10": 1.0, "mrr": 1.0}, "dense": None, "hybrid": None},
        "index": {"documents": 1, "documents_without_chunks": 0, "chunks": 1, "levels": {},
                  "chunk_tokens_p50": 1, "chunk_tokens_p90": 1, "chunk_tokens_max": 1,
                  "text_tokens": 1, "embedded_tokens": 3, "embed_cost_full_usd": 0.0},
        "reused_index": False, "seconds_per_question": 0.1, "seconds": 1.0,
        "embedding": {"cache": {}, "tokens_this_run": 0, "spent_this_run_usd": 0.0,
                      "ledger_total_usd": 0.0},
        "per_question": [{"question_id": "q_0001", "gold": 1, "gold_ranks": {"bm25": [1]}}],
    }  # fmt: skip
    text = render_markdown(result)
    assert "| dense | n/a" in text and "| q_0001 | 1 | 1 |" in text
    assert "| mode | recall@5 | recall@10 | recall@20 | q_recall@5 | q_recall@10 |" in text
    assert "| bm25 | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 | 0.500 | 1.000 | 1.000 |" in text
    assert "(pooled)" in text and "(per question)" in text
    assert history_metrics(result).startswith("bm25 recall@5=1.000")
    assert "q_recall@10=0.500" in history_metrics(result)
    assert headline(result) == ("bm25_q_recall@10", 0.5)  # no dense side: bm25
    hybrid = {"recall@10": 0.9, "q_recall@10": 0.7}
    with_dense = {**result, "dense": True, "scores": {**result["scores"], "hybrid": hybrid}}
    assert headline(with_dense) == ("hybrid_q_recall@10", 0.7)
    # an old report has no weights: it was plain RRF
    assert "RRF k = 60, weights bm25 1 / dense 1," in text
    weighted = {**result, "retriever": {"top_chunks": 50, **Fusion(0.5, 2.0, 20).describe()}}
    assert "RRF k = 20, weights bm25 0.5 / dense 2," in render_markdown(weighted)
    # Codex review 2026-10-16 finding 3: a report from before per-question recall (no
    # q_recall keys and no weights) failed with KeyError; its missing cells read n/a.
    old_scores = {k: v for k, v in result["scores"]["bm25"].items() if not k.startswith("q_")}
    old = {**result, "scores": {**result["scores"], "bm25": old_scores}}
    assert "| bm25 | 1.000 | 1.000 | 1.000 | n/a | n/a | n/a | 1.000 | 1.000 |" in render_markdown(
        old
    )


def test_the_cli_scores_either_a_variant_or_the_production_index() -> None:
    assert parse_args(["--production"]).production
    assert parse_args(["--chunking", "recursive"]).chunking == "recursive"
    for argv in ([], ["--production", "--chunking", "whole"]):
        with pytest.raises(SystemExit):
            parse_args(argv)


def test_the_fields_flag_needs_a_variant_with_its_header() -> None:
    assert parse_args(["--chunking", "recursive", "--fields"]).fields
    assert not parse_args(["--chunking", "recursive"]).fields
    for argv in (["--production", "--fields"], ["--chunking", "whole", "--fields", "--no-header"]):
        with pytest.raises(SystemExit):
            parse_args(argv)


def test_fusion_flags_default_to_config_and_reject_negative_values() -> None:
    plain = parse_args(["--production"])
    assert (plain.bm25_weight, plain.dense_weight, plain.rrf_k) == (
        RRF_BM25_WEIGHT, RRF_DENSE_WEIGHT, RRF_K,
    )  # fmt: skip
    argv = ["--chunking", "recursive", "--bm25-weight", "0.5", "--dense-weight", "2"]
    tuned = parse_args([*argv, "--rrf-k", "20"])
    assert (tuned.bm25_weight, tuned.dense_weight, tuned.rrf_k) == (0.5, 2.0, 20)
    for bad in (["--bm25-weight", "-1"], ["--dense-weight", "-0.5"], ["--rrf-k", "-1"]):
        with pytest.raises(SystemExit):
            parse_args(["--production", *bad])


@pytest.mark.parametrize("flag", ["--bm25-weight", "--dense-weight"])
@pytest.mark.parametrize("value", ["nan", "inf", "1e999", "-inf"])
def test_fusion_flags_reject_non_finite_weights(flag: str, value: str) -> None:
    # Codex review 2026-10-16 finding 1: these passed the `< 0` check and reached the run.
    with pytest.raises(SystemExit):
        parse_args(["--production", flag, value])


def test_fusion_suffix_names_only_what_differs_from_the_defaults() -> None:
    assert (RRF_BM25_WEIGHT, RRF_DENSE_WEIGHT, RRF_K) == (1.0, 1.0, 60)  # the cases assume it
    assert Fusion().suffix == ""
    assert Fusion(0.5, 1.0, 20).suffix == "-w0.5x1-k20"
    assert Fusion(1.0, 2.0).suffix == "-w1x2"
    assert Fusion(rrf_k=90).suffix == "-k90"
    assert Fusion(0.5, 1.0, 20).describe() == {"rrf_k": 20, "bm25_weight": 0.5, "dense_weight": 1.0}


def test_fusion_suffix_tells_close_weights_apart_and_names_zero_once() -> None:
    # Codex review 2026-10-16 finding 2: `:g` keeps six significant digits, so weights that
    # rank differently shared one name; -0.0 (accepted: it is >= 0) was named "-0".
    assert Fusion(1.0000001, 1.0).suffix != Fusion(1.0000002, 1.0).suffix
    assert Fusion(1.0000001, 1.0).suffix == "-w1.0000001x1"
    assert Fusion(-0.0, 1.0).suffix == Fusion(0.0, 1.0).suffix == "-w0x1"
    # weights that read back exactly keep their short names (the history rows use them)
    assert Fusion(0.25, 1.0).suffix == "-w0.25x1" and Fusion(1.0, 2.0).suffix == "-w1x2"


def test_a_screened_index_is_never_reused_for_other_gold_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # B5 review finding 10: the screened variant's manifest did not record which documents
    # it kept, so a run on another question set reused an index without its gold documents.
    from corpusqa.config import Settings
    from eval import retrieval
    from tests.unit.test_ingest_pipeline import FIXTURE

    monkeypatch.setattr(retrieval, "EVAL_CACHE_DIR", tmp_path / "eval")
    monkeypatch.setattr(retrieval, "EVAL_SCREEN_SAMPLE_DOCS", 0)  # keep the gold docs only
    settings = Settings(corpus_dir=FIXTURE, port=8080, index_dir=tmp_path / "index",
                        embed_cache_db=tmp_path / "cache.sqlite")  # fmt: skip
    args = parse_args(["--chunking", "recursive", "--screen"])

    def indexed(gold: str) -> tuple[bool, set[str]]:
        questions = [retrieval.Question("q_0001", "invented", frozenset({gold}))]
        target = retrieval.variant_index(args, settings, None, questions)
        assert isinstance(target, retrieval.Target)
        assert isinstance(target.chunks, retrieval.SqliteChunkReader)
        return target.reused, set(target.chunks.chunk_doc_ids().values())

    assert indexed("fx-tkt-102") == (False, {"fx-tkt-102"})
    assert indexed("fx-tkt-102") == (True, {"fx-tkt-102"})  # same documents: reused
    assert indexed("acct:77") == (False, {"acct:77"})  # other documents: rebuilt
