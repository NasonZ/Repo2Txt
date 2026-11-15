# BCY Budget Curve

- Generated at: `2026-11-15T20:20:32+00:00`
- Corpus manifest: `/tmp/tmpmp8wjdhg/corpus_manifest.jsonl`
- Tokenizer: `regex_code_tokenizer_v1`
- Protocol: file-deduplicate ranked results, render canonical corpus file text with a path header, greedily pack by rank, and prefix-truncate at the budget boundary.
- Canonical coverage: a gold file counts when at least one non-header content token is packed.
- Sensitivity coverage: at threshold tau, require at least min(tau, available file-content tokens).
- Caveat: this report uses released corpus `kind=file` text and stored top-20 file lists.

## Overall Curve

| Rank | Method | Family | Samples | BCY@4k | BCY@8k | BCY@16k | BCY@32k |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | repo2txt-llm | repo2txt | 345 | 0.4874 | 0.6087 | 0.6321 | 0.6481 |
| 2 | repo2txt-bm25 | repo2txt | 345 | 0.1457 | 0.2664 | 0.3671 | 0.4256 |
| 3 | repo2txt-bm25-plain | repo2txt | 345 | 0.1546 | 0.2357 | 0.3201 | 0.4758 |

## code2test

| Rank | Method | Family | Samples | BCY@4k | BCY@8k | BCY@16k | BCY@32k |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | repo2txt-llm | repo2txt | 106 | 0.2469 | 0.4104 | 0.4151 | 0.4340 |
| 2 | repo2txt-bm25-plain | repo2txt | 106 | 0.0252 | 0.0991 | 0.2126 | 0.3277 |
| 3 | repo2txt-bm25 | repo2txt | 106 | 0.0000 | 0.0031 | 0.0173 | 0.0739 |

## comment2context

| Rank | Method | Family | Samples | BCY@4k | BCY@8k | BCY@16k | BCY@32k |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | repo2txt-llm | repo2txt | 80 | 0.3208 | 0.4354 | 0.4688 | 0.5062 |
| 2 | repo2txt-bm25-plain | repo2txt | 80 | 0.1521 | 0.1938 | 0.2896 | 0.3542 |
| 3 | repo2txt-bm25 | repo2txt | 80 | 0.1021 | 0.1729 | 0.2354 | 0.2938 |

## trace2code

| Rank | Method | Family | Samples | BCY@4k | BCY@8k | BCY@16k | BCY@32k |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | repo2txt-llm | repo2txt | 101 | 0.8647 | 0.9373 | 0.9587 | 0.9637 |
| 2 | repo2txt-bm25 | repo2txt | 101 | 0.3300 | 0.6238 | 0.8201 | 0.8944 |
| 3 | repo2txt-bm25-plain | repo2txt | 101 | 0.2178 | 0.3267 | 0.3845 | 0.6535 |

## edit2ripple

| Rank | Method | Family | Samples | BCY@4k | BCY@8k | BCY@16k | BCY@32k |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | repo2txt-llm | repo2txt | 58 | 0.5000 | 0.6379 | 0.6853 | 0.6853 |
| 2 | repo2txt-bm25-plain | repo2txt | 58 | 0.2845 | 0.3851 | 0.4468 | 0.6049 |
| 3 | repo2txt-bm25 | repo2txt | 58 | 0.1509 | 0.2543 | 0.3994 | 0.4339 |

## Packing Diagnostics

| Method | Used@8k | Packed files@8k | Partial files@8k | Missing ranked files@8k |
|---|---:|---:|---:|---:|
| repo2txt-llm | 7981.4435 | 7.1942 | 0.9797 | 0.0000 |
| repo2txt-bm25 | 7974.2986 | 6.8986 | 0.9826 | 0.0000 |
| repo2txt-bm25-plain | 7999.9304 | 6.0261 | 0.9971 | 0.0000 |

## Minimum-content Threshold Sensitivity at 8k

| Method | tau=1 | tau=16 | tau=32 | tau=64 | tau=128 |
|---|---:|---:|---:|---:|---:|
| repo2txt-llm | 0.6087 | 0.6087 | 0.6087 | 0.6087 | 0.6087 |
| repo2txt-bm25 | 0.2664 | 0.2650 | 0.2650 | 0.2650 | 0.2650 |
| repo2txt-bm25-plain | 0.2357 | 0.2357 | 0.2357 | 0.2357 | 0.2357 |
