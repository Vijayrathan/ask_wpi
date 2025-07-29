# AskWPI Model Evaluation System

A comprehensive evaluation system for the AskWPI chatbot models (RAG, ReACT, and Fine-tuned LLM).

## Quick Start

```bash
# Run evaluation on all models
python3 run_full_evaluation.py
```

## What's Included

- **evaluation/**: Core evaluation system

  - `metrics.py`: Evaluation metrics (Exact Match, Semantic Accuracy, F1 Score, Multi-turn Consistency)
  - `evaluator.py`: Main evaluator that runs tests on all models
  - `test_data.json`: Custom test data with 8 WPI-specific queries
  - `README.md`: Detailed documentation

- **run_full_evaluation.py**: Main script to run complete evaluation

## Models Evaluated

- **RAG**: Retrieval-Augmented Generation
- **ReACT**: Reasoning and Acting agent
- **Fine-tuned LLM**: Fine-tuned Mistral model

## Metrics

- **Exact Match**: Binary text matching
- **Semantic Accuracy**: Cosine similarity using sentence embeddings
- **F1 Score**: Keyword overlap scoring
- **Multi-turn Consistency**: Consistency across conversation turns

## Output

Results are saved to `evaluation_results/`:

- `all_evaluation_results.json` - Complete results
- `model_comparison.csv` - Comparison table
- `summary_statistics.json` - Summary statistics
- `[model]_evaluation_results.json` - Individual model results

## Requirements

- `MISTRAL_API_KEY` environment variable
- Python packages: `sentence-transformers`, `nltk`, `scikit-learn`, `pandas`, `numpy`
