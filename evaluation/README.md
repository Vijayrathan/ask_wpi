# AskWPI Model Evaluation System

A simple evaluation system for the AskWPI chatbot models (RAG, ReACT, and Fine-tuned LLM).

## Usage

```bash
# Run complete evaluation on all models
python run_full_evaluation.py
```

## Metrics

- **Exact Match**: Binary score for exact text matching
- **Semantic Accuracy**: Cosine similarity using sentence embeddings
- **F1 Score**: Keyword overlap-based scoring
- **Multi-turn Consistency**: Consistency across conversation turns

## Output

Results are saved to `evaluation_results/` directory:

- `all_evaluation_results.json` - Complete results
- `model_comparison.csv` - Comparison table
- `summary_statistics.json` - Summary statistics
- `[model]_evaluation_results.json` - Individual model results

## Requirements

- `MISTRAL_API_KEY` environment variable set
- `sentence-transformers`, `nltk`, `scikit-learn`, `pandas`, `numpy` installed
- `finetuned_llm/dataset.jsonl` file exists
