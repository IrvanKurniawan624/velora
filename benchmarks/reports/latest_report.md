# Velora Benchmark Report — MODE=hybrid

*Generated on: 2026-07-13T10:22:59+00:00*

## Summary

| Metric | Value |
| :--- | :--- |
| Accuracy | 57.4% (13/27) |
| Total input tokens (est.) | 1312 |
| Total output tokens (est.) | 1022 |
| Remote tokens (hybrid) | 0 |
| Execution time | 445.8s |

## Category Breakdown

| Category | Accuracy | Tasks | Input tokens | Output tokens |
| :--- | :---: | :---: | :---: | :---: |
| factual | 75.0% | 4 | 78 | 126 |
| math | 75.0% | 4 | 109 | 10 |
| sentiment | 75.0% | 4 | 218 | 176 |
| summarisation | 50.0% | 3 | 317 | 164 |
| ner | 0.0% | 3 | 155 | 30 |
| debugging | 0.0% | 3 | 189 | 255 |
| logic | 66.7% | 3 | 104 | 74 |
| codegen | 100.0% | 3 | 142 | 187 |

## Per-Task Details

| Task ID | Category | Score | Input | Output | Reason |
| :--- | :--- | :---: | :---: | :---: | :--- |
| factual-01 | factual | 1.00 | 19 | 32 | Matched 3/3 keywords |
| factual-02 | factual | 0.33 | 23 | 38 | Matched 1/3 keywords |
| factual-03 | factual | 1.00 | 17 | 20 | Matched 2/2 keywords |
| factual-04 | factual | 0.67 | 19 | 36 | Matched 2/3 keywords |
| math-01 | math | 1.00 | 30 | 4 | Model: 105 | Expected: 105 |
| math-02 | math | 1.00 | 21 | 1 | Model: 150 | Expected: 150 |
| math-03 | math | 0.00 | 23 | 4 | Model: 108 | Expected: 144 |
| math-04 | math | 1.00 | 35 | 1 | Model: 120 | Expected: 120 |
| sentiment-01 | sentiment | 1.00 | 59 | 42 | Sentiment Match: True, Has Reason: True |
| sentiment-02 | sentiment | 1.00 | 60 | 51 | Sentiment Match: True, Has Reason: True |
| sentiment-03 | sentiment | 0.00 | 40 | 41 | Sentiment Match: False, Has Reason: True |
| sentiment-04 | sentiment | 1.00 | 59 | 42 | Sentiment Match: True, Has Reason: True |
| summarisation-01 | summarisation | 0.50 | 107 | 56 | Found 1 bullets (expected 2) |
| summarisation-02 | summarisation | 0.50 | 106 | 57 | Found 1 bullets (expected 2) |
| summarisation-03 | summarisation | 0.50 | 104 | 51 | Found 1 bullets (expected 2) |
| ner-01 | ner | 0.00 | 48 | 9 | Mismatch in PERSON, Mismatch in ORG, Mismatch in LOC |
| ner-02 | ner | 0.00 | 56 | 12 | Mismatch in PERSON, Mismatch in ORG, Mismatch in LOC, Mismatch in DATE |
| ner-03 | ner | 0.00 | 51 | 9 | Mismatch in PERSON, Mismatch in ORG, Mismatch in LOC |
| debugging-01 | debugging | 0.00 | 79 | 86 | Assertion failed: invalid syntax (<string>, line 1) |
| debugging-02 | debugging | 0.00 | 42 | 84 | Assertion failed: invalid syntax (<string>, line 1) |
| debugging-03 | debugging | 0.00 | 68 | 85 | Assertion failed: invalid syntax (<string>, line 1) |
| logic-01 | logic | 0.00 | 37 | 22 | Expected 'charlie' in answer |
| logic-02 | logic | 1.00 | 40 | 24 | Expected 'sam' in answer |
| logic-03 | logic | 1.00 | 27 | 28 | Expected 'yes' in answer |
| codegen-01 | codegen | 1.00 | 54 | 27 | Passes all is_palindrome assertions |
| codegen-02 | codegen | 1.00 | 45 | 89 | Passes all second_largest assertions |
| codegen-03 | codegen | 1.00 | 43 | 71 | Passes all is_prime assertions |
