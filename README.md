# Financial Risk Intelligence System

A production-style 4-agent AI pipeline for automated financial risk 
classification and explanation generation.

## Overview
This system processes financial news articles and classifies companies 
into High / Medium / Low risk categories using NLP and machine learning, 
then generates plain-English explanations using a local LLM.

## Pipeline Architecture
| Agent | Role | Technique |
|-------|------|-----------|
| Agent 1 — Retriever | Semantic search over 35,191 news articles | Sentence Embeddings + Cosine Similarity |
| Agent 2 — Analyzer | Risk classification (High/Medium/Low) | Logistic Regression |
| Agent 3 — Generator | Plain-English explanation | RAG + Qwen3-0.6B LLM |
| Agent 4 — Critic | QA validation of LLM output | Rule-based checks |

## Results
- Macro F1: 0.47 on imbalanced 3-class dataset
- High Risk recall: 0.53 (prioritized to minimize missed risk detections)
- Medium Risk F1: 0.64

## Tech Stack
- Python, scikit-learn, HuggingFace Transformers
- Sentence Transformers (all-MiniLM-L6-v2)
- Qwen3-0.6B (local LLM)
- Pandas, NumPy, Matplotlib

## Key Features
- Handles severe class imbalance (12.3% High Risk) using balanced class weights
- Fixed prompt-echo bug in LLM decoding (output[0][prompt_len:])
- Critic Agent validates length, risk-keyword consistency, and repetition

## How to Run
```bash
pip install sentence-transformers transformers scikit-learn pandas numpy
jupyter notebook ML_Project1_fixed.ipynb
```

## Dataset
- `cleaned_news_dataset.csv` — 35,191 financial news articles
- `cleaned_sentiment_data.csv` — 4,149 labelled financial sentences
