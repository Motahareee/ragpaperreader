# PaperChat retrieval benchmark: results

Hardware caveat: measured in an emulated Linux ARM64 sandbox, not the target Apple Silicon Mac (no Metal acceleration here). Numbers are trustworthy for *relative* comparisons between the variants below, but won't match absolute latency on real hardware -- re-measure there before citing absolute numbers.

Corpus: 2 papers (`attention.pdf`, `adam.pdf`), 15 pages / ~35 chunks each, 71 chunks total.

## Accuracy

Two separate sets, because the original golden set was curated to only include questions that already worked with cosine -- it can't show accuracy *differences* between methods.

**Verified golden set** (`tests/golden/qa_set.json`, n=8, hand-verified against `attention.pdf` only):

| Variant | Recall@5 |
|---|---|
| MiniLM + cosine | 1.00 (8/8) |
| bge-small + cosine | 1.00 (8/8) |
| MiniLM + hybrid (cosine+BM25) | 1.00 (8/8) |

**Adversarial hard set** (`benchmark/hard_qa_set.json`, n=6, spans both PDFs, includes cases picked specifically because they miss with plain cosine):

| Variant | Recall@5 |
|---|---|
| MiniLM + cosine | 0.67 (4/6) |
| bge-small + cosine | 0.67 (4/6) |
| **MiniLM + hybrid (cosine+BM25 via RRF)** | **0.83 (5/6)** |

Finding: hybrid retrieval measurably helped -- it recovered a keyword-heavy miss ("What dataset was used for the English to German translation task?") that pure cosine missed on both encoders, consistent with the predicted failure mode (dense embeddings blur past exact terms that keyword matching catches directly). bge-small didn't outperform MiniLM on this sample; same overall rate, different specific miss (Pdrop dropout value instead of the dataset question).

## Latency

| Stage | MiniLM | bge-small |
|---|---|---|
| Indexing throughput | 4.9-6.8 pages/s | 2.2-3.4 pages/s (~2x slower, roughly matches bge-small having ~2x MiniLM's transformer layers) |
| Single-query embed | 4.8ms avg | 203ms avg (~42x slower -- disproportionate to the ~2x batch-throughput gap; flagged as an open question, likely call-overhead specific to this GGUF, not a fundamental architecture cost) |
| Search (numpy dot product) | 0.10ms avg | 0.67ms avg |
| Hybrid (embed + BM25 + RRF fuse) | 15.5ms avg | -- |

## Memory

| Checkpoint | MiniLM | bge-small |
|---|---|---|
| Model file size | 23.8 MB | 35.1 MB |
| Peak RSS after model load | 130.9 MB | 142.8 MB |
| Peak RSS after indexing 2 papers | 193.3 MB | 205.1 MB |

Both are small relative to the ~4.7GB instruct LLM, which dominates real-world memory use once loaded for answering.

## Scalability (numpy brute-force search vs. corpus size)

Synthetic 384-dim unit vectors, mean of 20 queries per size:

| n_vectors | mean latency | p95 latency |
|---|---|---|
| 10 | 0.003ms | 0.007ms |
| 100 | 0.009ms | 0.010ms |
| 1,000 | 0.071ms | 0.076ms |
| 10,000 | 1.91ms | 9.29ms |
| 100,000 | 14.2ms | 22.1ms |
| 500,000 | 60.4ms | 77.8ms |

Even at 500K vectors -- far beyond what any realistic "folder of papers" would contain -- brute-force search stays under 80ms p95. This is a direct, measured answer to "should we use a real vector database": no, not remotely close to the scale this tool operates at. The architectural choice (numpy over Chroma/FAISS, made originally for packaging-simplicity reasons) also holds up on pure performance grounds.

## Reproducing

```bash
python scripts/fetch_embed_model.py          # MiniLM (bundled default)
python benchmark/run_baseline.py             # MiniLM + cosine, all axes
python benchmark/run_bge.py                  # bge-small + cosine (downloads its own GGUF)
python benchmark/run_hybrid.py               # MiniLM + hybrid, on the verified golden set
python benchmark/run_hard_comparison.py      # all 3 variants on the adversarial hard set
```

Raw JSON reports land in `benchmark/results/`.
