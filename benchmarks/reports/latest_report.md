# Velora Benchmark Report — Zero vs Hybrid

*Generated on: 2026-07-13*

## Summary

| Mode | Accuracy | Remote tokens | Est. local tokens (in+out) | Time |
| :--- | :---: | :---: | :---: | :---: |
| **Zero** | 57.4% (13/27) | 0 | 2,357 | 481.1s |
| **Hybrid** | 74.1% (20/27) | 1,331 | 2,392 | 420.4s |

Hybrid gained **+16.7 pp** accuracy at a cost of **1,331 remote tokens**.

## Category Breakdown

| Category | Zero accuracy | Hybrid accuracy | Tasks |
| :--- | :---: | :---: | :---: |
| factual | 75.0% | 75.0% | 4 |
| math | 75.0% | 75.0% | 4 |
| sentiment | 75.0% | 75.0% | 4 |
| summarisation | 50.0% | 100.0% | 3 |
| ner | 0.0% | 100.0% | 3 |
| debugging | 0.0% | 0.0% | 3 |
| logic | 66.7% | 66.7% | 3 |
| codegen | 100.0% | 100.0% | 3 |

## Notes

- Hybrid escalates tasks whose local confidence is ≤ 0.35. NER and summarisation (local conf 0.25) are sent to Fireworks and jump from 0% to 100% on those categories.
- Debugging remains 0%: the local model produces syntactically invalid code, and the remote model also fails to pass the test assertions for these 3 tasks.
- Codegen stays at 100%; math/factual/sentiment stay unchanged; the hybrid token spend is on NER and summarisation.

## Full per-mode reports

- Zero: `benchmarks/reports/report_20260713_113925.md`
- Hybrid: `benchmarks/reports/report_20260713_113043.md`
