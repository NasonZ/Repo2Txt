# BCY Budget Curve

- Generated at: `2026-11-15T22:37:46+00:00`
- Corpus manifest: `/tmp/tmpett9dx2z/corpus_manifest.jsonl`
- Tokenizer: `regex_code_tokenizer_v1`
- Protocol: file-deduplicate ranked results, render canonical corpus file text with a path header, greedily pack by rank, and prefix-truncate at the budget boundary.
- Canonical coverage: a gold file counts when at least one non-header content token is packed.
- Sensitivity coverage: at threshold tau, require at least min(tau, available file-content tokens).
- Caveat: this report uses released corpus `kind=file` text and stored top-20 file lists.

## Overall Curve

| Rank | Method | Family | Samples | BCY@4k | BCY@8k | BCY@16k | BCY@32k |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | arb-llm-haiku | repo2txt | 345 | 0.5036 | 0.6295 | 0.6488 | 0.6638 |
| 2 | arb-llm-deepseek | repo2txt | 345 | 0.4802 | 0.6126 | 0.6394 | 0.6539 |
| 3 | repo2txt-llm | repo2txt | 345 | 0.4874 | 0.6087 | 0.6321 | 0.6481 |
| 4 | arb-llm-sonnet | repo2txt | 345 | 0.4691 | 0.5341 | 0.5493 | 0.5691 |
| 5 | repo2txt-bm25 | repo2txt | 345 | 0.1457 | 0.2664 | 0.3671 | 0.4256 |

## code2test

| Rank | Method | Family | Samples | BCY@4k | BCY@8k | BCY@16k | BCY@32k |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | arb-llm-haiku | repo2txt | 106 | 0.2940 | 0.4513 | 0.4654 | 0.4748 |
| 2 | arb-llm-deepseek | repo2txt | 106 | 0.2657 | 0.4434 | 0.4528 | 0.4623 |
| 3 | repo2txt-llm | repo2txt | 106 | 0.2469 | 0.4104 | 0.4151 | 0.4340 |
| 4 | arb-llm-sonnet | repo2txt | 106 | 0.2233 | 0.2704 | 0.2846 | 0.3035 |
| 5 | repo2txt-bm25 | repo2txt | 106 | 0.0000 | 0.0031 | 0.0173 | 0.0739 |

## comment2context

| Rank | Method | Family | Samples | BCY@4k | BCY@8k | BCY@16k | BCY@32k |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | arb-llm-haiku | repo2txt | 80 | 0.3042 | 0.4500 | 0.4729 | 0.5188 |
| 2 | repo2txt-llm | repo2txt | 80 | 0.3208 | 0.4354 | 0.4688 | 0.5062 |
| 3 | arb-llm-deepseek | repo2txt | 80 | 0.2792 | 0.4125 | 0.4437 | 0.4854 |
| 4 | arb-llm-sonnet | repo2txt | 80 | 0.2188 | 0.3333 | 0.3458 | 0.3958 |
| 5 | repo2txt-bm25 | repo2txt | 80 | 0.1021 | 0.1729 | 0.2354 | 0.2938 |

## trace2code

| Rank | Method | Family | Samples | BCY@4k | BCY@8k | BCY@16k | BCY@32k |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | arb-llm-deepseek | repo2txt | 101 | 0.8795 | 0.9521 | 0.9785 | 0.9785 |
| 2 | arb-llm-sonnet | repo2txt | 101 | 0.8993 | 0.9439 | 0.9554 | 0.9604 |
| 3 | arb-llm-haiku | repo2txt | 101 | 0.8828 | 0.9422 | 0.9587 | 0.9637 |
| 4 | repo2txt-llm | repo2txt | 101 | 0.8647 | 0.9373 | 0.9587 | 0.9637 |
| 5 | repo2txt-bm25 | repo2txt | 101 | 0.3300 | 0.6238 | 0.8201 | 0.8944 |

## edit2ripple

| Rank | Method | Family | Samples | BCY@4k | BCY@8k | BCY@16k | BCY@32k |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | arb-llm-haiku | repo2txt | 58 | 0.5014 | 0.6580 | 0.6868 | 0.6868 |
| 2 | repo2txt-llm | repo2txt | 58 | 0.5000 | 0.6379 | 0.6853 | 0.6853 |
| 3 | arb-llm-deepseek | repo2txt | 58 | 0.4540 | 0.6063 | 0.6595 | 0.6710 |
| 4 | arb-llm-sonnet | repo2txt | 58 | 0.5144 | 0.5790 | 0.6063 | 0.6121 |
| 5 | repo2txt-bm25 | repo2txt | 58 | 0.1509 | 0.2543 | 0.3994 | 0.4339 |

## Packing Diagnostics

| Method | Used@8k | Packed files@8k | Partial files@8k | Missing ranked files@8k |
|---|---:|---:|---:|---:|
| arb-llm-haiku | 7990.9710 | 7.0493 | 0.9826 | 0.0000 |
| arb-llm-deepseek | 7985.3565 | 7.2812 | 0.9826 | 0.0000 |
| repo2txt-llm | 7981.4435 | 7.1942 | 0.9797 | 0.0000 |
| arb-llm-sonnet | 7978.4290 | 6.8667 | 0.9826 | 0.0000 |
| repo2txt-bm25 | 7974.2986 | 6.8986 | 0.9826 | 0.0000 |

## Minimum-content Threshold Sensitivity at 8k

| Method | tau=1 | tau=16 | tau=32 | tau=64 | tau=128 |
|---|---:|---:|---:|---:|---:|
| arb-llm-haiku | 0.6295 | 0.6295 | 0.6295 | 0.6295 | 0.6295 |
| arb-llm-deepseek | 0.6126 | 0.6126 | 0.6126 | 0.6126 | 0.6126 |
| repo2txt-llm | 0.6087 | 0.6087 | 0.6087 | 0.6087 | 0.6087 |
| arb-llm-sonnet | 0.5341 | 0.5341 | 0.5341 | 0.5341 | 0.5326 |
| repo2txt-bm25 | 0.2664 | 0.2650 | 0.2650 | 0.2650 | 0.2650 |
