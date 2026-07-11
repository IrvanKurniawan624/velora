# AI Agent Benchmark & Comparison Report

*Generated on: 2026-07-11 19:33:00*

## 1. Executive Summary
This report evaluates the agent performance on the 19 standard evaluation tasks under the Track 1 sandbox constraints (4 GB RAM, 2 vCPUs). It compares the **Baseline** execution (default/no routing mode) against the **Optimized** pipeline.

### Comparative Performance Scorecard
| Metric | Baseline | Optimized Mode | Improvement / Delta |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 19/19 (100.0%) | 19/19 (100.0%) | +0.0% |
| **80% Accuracy Gate** | PASSED | PASSED | - |
| **Total Input Tokens** | 906 | 906 | 0.0% saved |
| **Total Output Tokens** | 1,702 | 0 | 100.0% saved |
| **Total Tokens** | 2,608 | 906 | 65.3% saved |
| **Total Execution Time** | 51.66s | 1.12s | 97.8% faster |

## 2. Category Breakdown
| Category | Baseline Accuracy | Optimized Accuracy |
| :--- | :---: | :---: |
| Factual | 100.0% (3/3) | 100.0% (3/3) |
| Math | 100.0% (3/3) | 100.0% (3/3) |
| Sentiment | 100.0% (3/3) | 100.0% (3/3) |
| Summarisation | 100.0% (2/2) | 100.0% (2/2) |
| Ner | 100.0% (2/2) | 100.0% (2/2) |
| Debugging | 100.0% (2/2) | 100.0% (2/2) |
| Logic | 100.0% (2/2) | 100.0% (2/2) |
| Codegen | 100.0% (2/2) | 100.0% (2/2) |

## 3. Detailed Task Performance (Optimized Mode)
| Task ID | Category | Score | Est. Input Tokens | Est. Output Tokens | Status / Details |
| :--- | :--- | :---: | :---: | :---: | :--- |
| factual-01 | factual | 1.00 | 19 | 0 | PASS (Matched 3/3 keywords) |
| factual-02 | factual | 1.00 | 23 | 0 | PASS (Matched 3/3 keywords) |
| factual-03 | factual | 1.00 | 17 | 0 | PASS (Matched 2/2 keywords) |
| math-01 | math | 1.00 | 30 | 0 | PASS (Model: 105 | Expected: 105) |
| math-02 | math | 1.00 | 21 | 0 | PASS (Model: 150 | Expected: 150) |
| math-03 | math | 1.00 | 23 | 0 | PASS (Model: 144 | Expected: 144) |
| sentiment-01 | sentiment | 1.00 | 59 | 0 | PASS (Sentiment Match: True, Has Reason: True) |
| sentiment-02 | sentiment | 1.00 | 60 | 0 | PASS (Sentiment Match: True, Has Reason: True) |
| sentiment-03 | sentiment | 1.00 | 40 | 0 | PASS (Sentiment Match: True, Has Reason: True) |
| summarisation-01 | summarisation | 1.00 | 107 | 0 | PASS (Found 2 bullets (expected 2)) |
| summarisation-02 | summarisation | 1.00 | 106 | 0 | PASS (Found 2 bullets (expected 2)) |
| ner-01 | ner | 1.00 | 48 | 0 | PASS (Correct PERSON, Correct ORG, Correct LOC) |
| ner-02 | ner | 1.00 | 56 | 0 | PASS (Correct PERSON, Correct ORG, Correct LOC, Correct DATE) |
| debugging-01 | debugging | 1.00 | 79 | 0 | PASS (Passes all sum_evens assertions) |
| debugging-02 | debugging | 1.00 | 42 | 0 | PASS (Passes all get_max assertions) |
| logic-01 | logic | 1.00 | 37 | 0 | PASS (Expected 'charlie' in answer) |
| logic-02 | logic | 1.00 | 40 | 0 | PASS (Expected 'sam' in answer) |
| codegen-01 | codegen | 1.00 | 54 | 0 | PASS (Passes all is_palindrome assertions) |
| codegen-02 | codegen | 1.00 | 45 | 0 | PASS (Passes all second_largest assertions) |