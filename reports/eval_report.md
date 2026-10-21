# RevioIQ Evaluation Report

Evaluation results for the RevioIQ Review Intelligence Engine PoC.
*Generated at: 2026-10-21 13:18:45 UTC*

## 1. Executive Summary & Headline Metrics

| Component | Metric | Score | Dataset & Conditions |
|---|---|---|---|
| **Sentiment Baseline** | Accuracy | **76.70%** | 2,000 Yelp Full Reviews (seeded test split) |
| **Sentiment Baseline** | Macro-F1 | **0.7264** | 3-class (negative / neutral / positive) |
| **Crisis Detector** | Recall (Priority) | **100.00%** | Synthetic benchmark (40 realistic reviews) |
| **Crisis Detector** | Precision | **93.75%** | Synthetic benchmark (40 realistic reviews) |
| **Crisis Detector** | F1 Score | **0.9677** | Synthetic benchmark (40 realistic reviews) |
| **Gemini Zero-Shot** | Accuracy | **30.00%** | Sample n=50 (gemini-3.8-flash) vs baseline 80.00% |
| **Gemini Zero-Shot** | Macro-F1 | **0.2735** | Sample n=50 (gemini-3.8-flash) vs baseline 0.7483 |
| **Reply Guardrails** | Validator Pass Rate | **100.00%** | 40 synthetic drafts checked by pure-Python validator |
| **Security Defense** | Injection Resistance | **100.00%** | 4/4 injection attacks neutralized |

---

## 2. Sentiment Baseline (TF-IDF + Logistic Regression)

- **Dataset**: Seeded subset of 20,000 train / 2,000 test reviews from `yelp_review_full`.
- **Label Mapping**: 1-2 stars -> `negative`, 3 stars -> `neutral`, 4-5 stars -> `positive`.
- **Test Accuracy**: 76.70%
- **Macro-F1**: 0.7264

### Confusion Matrix (Test Split: 2,000 Reviews)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 702 | 119 | 35 |
| **Neutral** | 80 | 212 | 80 |
| **Positive** | 35 | 117 | 620 |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Negative | 0.8592 | 0.8201 | 0.8392 | 856.0 |
| Neutral | 0.4732 | 0.5699 | 0.5171 | 372.0 |
| Positive | 0.8435 | 0.8031 | 0.8228 | 772.0 |
| **Macro Avg** | 0.7253 | 0.7310 | 0.7264 | 2000.0 |
| **Weighted Avg** | 0.7814 | 0.7670 | 0.7730 | 2000.0 |

### Error Analysis: 10 Failure Cases

Analysis of misclassifications reveals that the primary challenge is the **neutral category (3-star reviews)**, which frequently combines praise ('great cocktails') with critiques ('unbearably loud').

#### Case #1
- **Review Snippet**: *"Kabuto is your run-of-the-mill Japanese Steakhouse. Different stations with chefs slinging shrimp tails around the communal dining areas like it's a lunchtime magic show. Always a ..."*
- **True Label**: `neutral` | **Predicted Label**: `positive` (Confidence: 0.72, True Class Prob: 0.22)
- **Root Cause**: neutral boundary ambiguity (3-star rating variance)

#### Case #2
- **Review Snippet**: *"Visiting here for 10 days and staying in a condo\nDriving around and found this place\nAte here 2x ( Lunch + Dinner)\nFood was fresh + Tasty\nSushi was great for the price\nShrimp ..."*
- **True Label**: `positive` | **Predicted Label**: `neutral` (Confidence: 0.57, True Class Prob: 0.36)
- **Root Cause**: neutral boundary ambiguity (3-star rating variance)

#### Case #9
- **Review Snippet**: *"Went for lunch to celebrate a grinding training finally ending. Burgers and drinks were good. I wish I had time to sit there and savor it. \n\nI will go again with kids which is th..."*
- **True Label**: `neutral` | **Predicted Label**: `positive` (Confidence: 0.50, True Class Prob: 0.35)
- **Root Cause**: neutral boundary ambiguity (3-star rating variance)

#### Case #10
- **Review Snippet**: *"The drunken noodles..... Get them now. \n\nThe restaurant owner is very nice and you can tell he loves what he does. Now don't let the layout throw you. It may not be the fanciest ..."*
- **True Label**: `positive` | **Predicted Label**: `neutral` (Confidence: 0.44, True Class Prob: 0.38)
- **Root Cause**: neutral boundary ambiguity (3-star rating variance)

#### Case #16
- **Review Snippet**: *"Cirque du Soleil prices without the Cirque du Soleil talent.  I kept waiting to be amazed by people doing things with their bodies that I can't do.  That's why I pay you the big bu..."*
- **True Label**: `negative` | **Predicted Label**: `neutral` (Confidence: 0.49, True Class Prob: 0.20)
- **Root Cause**: neutral boundary ambiguity (3-star rating variance)

#### Case #17
- **Review Snippet**: *"Enough variety for me. Only reason I go to buffet - Shrimp, Oysters, Sushi and Salmon. They had all of these, so I'm happy. If you are looking for a quality buffet, skip this one ...."*
- **True Label**: `positive` | **Predicted Label**: `negative` (Confidence: 0.35, True Class Prob: 0.33)
- **Root Cause**: constructive critique inside overall positive review

#### Case #23
- **Review Snippet**: *"With a fun and upbeat bar area, this place is located in downtown Phoenix, close to the US Airways Center. It seemed that the family area was completely separated from the bar area..."*
- **True Label**: `positive` | **Predicted Label**: `neutral` (Confidence: 0.53, True Class Prob: 0.43)
- **Root Cause**: neutral boundary ambiguity (3-star rating variance)

#### Case #25
- **Review Snippet**: *"What do you do when Coca is packed? Well, sadly, the answer for us didn't lie at Mauramori, as much as we've tried over the past year. \n\nThe food here is fine, but that's really ..."*
- **True Label**: `negative` | **Predicted Label**: `neutral` (Confidence: 0.50, True Class Prob: 0.43)
- **Root Cause**: neutral boundary ambiguity (3-star rating variance)

#### Case #29
- **Review Snippet**: *"We live nearby and thought it would be great to try a local dental office.  Both times my husband and I went in the dental hygienist talked, and talked, and talked to the point of ..."*
- **True Label**: `negative` | **Predicted Label**: `positive` (Confidence: 0.51, True Class Prob: 0.37)
- **Root Cause**: subtle sarcasm or praise mixed with severe complaint

#### Case #32
- **Review Snippet**: *"I like the reasonably priced, reliable and tasty black bean burrito (the \"Wilmar\") with its fresh avocado inside.  Too much noise and hubbub though to eat it there during the rus..."*
- **True Label**: `neutral` | **Predicted Label**: `positive` (Confidence: 0.49, True Class Prob: 0.46)
- **Root Cause**: neutral boundary ambiguity (3-star rating variance)

---

## 3. Crisis Detector Performance (Synthetic Benchmark)

Evaluated on the synthetic multi-branch benchmark dataset (40 realistic reviews across 5 branches).

- Note: Data is explicitly synthetic and curated for comprehensive evaluation.
- **Precision**: 93.75%
- **Recall (Safety Priority)**: **100.00%**
- **F1 Score**: 0.9677

### Category Breakdown

| Crisis Category | Ground Truth Count | Detected Count | Category Recall |
|---|---|---|---|
| `food_poisoning` | 3 | 3 | **100.0%** |
| `legal_threat` | 3 | 3 | **100.0%** |
| `discrimination` | 3 | 3 | **100.0%** |
| `fraud_accusation` | 3 | 3 | **100.0%** |
| `media_threat` | 3 | 3 | **100.0%** |

---

## 4. LLM Zero-Shot & Guardrail Security Assessment

- **Model**: `gemini-3.8-flash`
- **Zero-Shot Sample Size**: 50 reviews randomly selected from test split
- **Gemini Accuracy**: **30.00%** (vs Baseline: 80.00%)
- **Gemini Macro-F1**: **0.2735** (vs Baseline: 0.7483)

### Guardrails & Prompt Injection Security

- **Validator Pass Rate**: **100.00%** of generated replies strictly satisfied financial, liability, PII, and length constraints.
- **Prompt Injection Attacks**: Tested 4 adversarial injection attacks attempting refund extortion and legal liability admissions.
- **Injection Resistance Rate**: **100.00%** (4/4 successfully neutralized).
- **Routing Safety**: 100% of crisis reviews and validator-rejected drafts were routed to the human escalation queue.
