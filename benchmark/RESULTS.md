# PaperChat retrieval benchmark: results

Hardware caveat: measured in an emulated Linux ARM64 sandbox, not the target Apple Silicon Mac (no Metal acceleration here). Numbers are trustworthy for *relative* comparisons between the variants below, but won't match absolute latency on real hardware -- re-measure there before citing absolute numbers.

Corpus: 5 papers (`attention.pdf`, `adam.pdf`, `bert.pdf`, `resnet.pdf`, `dropout.pdf`), 226 chunks total, spanning Transformers, optimization, language pretraining, computer vision, and regularization -- chosen for topic diversity to surface cross-document confusion effects a single-topic corpus can't.

## Accuracy

Two separate sets, because the original golden set was curated to only include questions that already worked with cosine -- it can't show accuracy *differences* between methods.

**Verified golden set** (`tests/golden/qa_set.json`, n=12, hand-verified doc+page for each answer, spans all 5 papers):

| Variant | Recall@5 |
|---|---|
| MiniLM + cosine | 1.00 (12/12) |
| bge-small + cosine | 0.83 (10/12) |
| MiniLM + hybrid (cosine+BM25) | 1.00 (12/12) |

**Adversarial hard set** (`benchmark/hard_qa_set.json`, n=11, cross-document + keyword-heavy cases verified to fail under plain cosine):

| Variant | Recall@5 |
|---|---|
| MiniLM + cosine | 0.27 (3/11) |
| bge-small + cosine | 0.64 (7/11) |
| **MiniLM + hybrid (cosine+BM25 via RRF)** | **0.64 (7/11)** |

Finding: no single variant dominates. bge-small trades golden-set accuracy (10/12 vs 12/12) for a large hard-set gain -- a real gap that only appeared once the corpus grew to 5 topically-varied papers, contradicting the general MTEB-benchmark expectation that bge-small should simply outperform MiniLM. **Hybrid retrieval is the strictly better overall choice**: it preserves MiniLM's full golden-set accuracy while matching bge-small's hard-set improvement, more than doubling the shipped system's hard-set recall (27% -> 64%).

A second finding came from the corpus expansion itself: one question that reliably worked at 2-paper scale ("How is positional information added to the input embeddings?") started missing once BERT's own positional-embedding discussion was added to the corpus -- moved into the hard set as a concrete example of accuracy degrading from topic overlap, invisible to any golden set built against a small corpus.

## Latency

| Stage | MiniLM | bge-small |
|---|---|---|
| Indexing throughput | 1.3-4.9 pages/s | 1.6-3.9 pages/s |
| Single-query embed (2-paper corpus) | 4.8ms | 203ms |
| Single-query embed (5-paper corpus) | 6.8ms | 14.6ms |
| Search (numpy dot product) | 0.46ms | 1.23ms |
| Hybrid (embed + BM25 + RRF fuse) | 7.8ms | -- |

bge-small's single-query latency swung ~14x between the two runs on what should be a corpus-size-independent measurement (203ms -> 14.6ms). This means the original "~42x slower than MiniLM" claim was very likely inflated by cold-start/measurement noise, not a stable architectural cost -- flagged explicitly rather than picking whichever number tells a cleaner story. Needs a proper warmup + repeated-trial protocol before either number is trustworthy.

## Memory

| Checkpoint | MiniLM | bge-small |
|---|---|---|
| Model file size | 23.8 MB | 35.1 MB |
| Peak RSS after model load | 130.8 MB | 142.5 MB |
| Peak RSS after indexing (5 papers, 226 chunks) | 194.3 MB | 205.9 MB |

Memory scaled negligibly from 2 to 5 papers (+1MB peak RSS) -- as expected for a numpy-array store, dominated by fixed model weights rather than corpus size. Both are small relative to the ~4.7GB instruct LLM, which dominates real-world memory use once loaded for answering.

## Scalability (numpy brute-force search vs. corpus size)

Synthetic 384-dim unit vectors, mean of 20 queries per size:

| n_vectors | mean latency | p95 latency |
|---|---|---|
| 10 | 0.003ms | 0.008ms |
| 1,000 | 0.074ms | 0.087ms |
| 10,000 | 1.74ms | 2.96ms |
| 100,000 | 14.4ms | 19.7ms |
| 500,000 | 55.7ms | 60.7ms |

Even at 500K vectors -- over 2,000x the actual 226-chunk corpus used here, and far beyond what any realistic "folder of papers" would contain -- brute-force search stays under 80ms p95. This is a direct, measured answer to "should we use a real vector database": no, not remotely close to the scale this tool operates at. The architectural choice (numpy over Chroma/FAISS, made originally for packaging-simplicity reasons) also holds up on pure performance grounds.

## Reproducing

```bash
python scripts/fetch_embed_model.py          # MiniLM (bundled default)
python benchmark/run_baseline.py             # MiniLM + cosine, all axes
python benchmark/run_bge.py                  # bge-small + cosine (downloads its own GGUF)
python benchmark/run_hybrid.py               # MiniLM + hybrid, on the verified golden set
python benchmark/run_hard_comparison.py      # all 3 variants on the adversarial hard set
```

Raw JSON reports land in `benchmark/results/`. A full write-up with a system architecture diagram is in `benchmark/report.html`.
