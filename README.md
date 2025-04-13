# FinBERT for Financial Sentiment Analysis

This project fine-tunes a domain-specific transformer model, FinBERT, to classify sentiment in financial news headlines. It benchmarks performance against traditional NLP baselines and the zero-shot FinBERT model, while providing actionable insights through company-level sentiment aggregation, trend analysis, and volatility metrics.

---

## Problem Statement

Financial text is highly nuanced. The same phrase can carry different implications depending on context and macroeconomic environment. General-purpose NLP models often fail to capture this subtletly. This project investigates whether domain adaptation through fine-tuning FinBERT improves sentiment classification performance compared to traditional methods and zero-shot transformer inference.

---

## Dataset

- **Source**: [Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank)
- **Size**: ~2,200 annotated financial sentences
- **Sentiment Classes**: `positive`, `neutral`, `negative`
- **Split Used**: `sentences_allagree` (100% annotator agreement)

---

## Models Evaluated

| Model                        | Accuracy | Description |
|-----------------------------|----------|-------------|
| TF-IDF + Logistic Regression | 87.0%    | Fast, interpretable, bag-of-words baseline |
| Pretrained FinBERT          | 91.7%    | Transformer without domain-specific adaptation |
| Fine-Tuned FinBERT          | 95.1%    | FinBERT retrained on task-specific labels using Trainer API |

---

## Methodology

1. **Preprocessing**: Cleaning, label encoding, tokenization
2. **Baseline Modeling**: TF-IDF + Logistic Regression
3. **Transformer Inference**: Pretrained FinBERT (zero-shot predictions)
4. **Fine-Tuning**: Hugging Face `Trainer` API with full model adaptation
5. **Evaluation**: Accuracy, confusion matrix, F1 scores
6. **Analysis**: 
   - Entity-level sentiment aggregation
   - Sentiment volatility scoring
   - Time-based trend analysis
   - Word cloud generation for interpretability

---

## Sentiment Volatility by Company

To highlight companies with inconsistent media tone, we computed a volatility score based on the standard deviation of sentiment predictions per entity.

| Company           | Volatility Score |
|------------------|------------------|
| Okmetic Oyj       | 1.41  
| Componenta Oyj    | 1.15  
| Suominen Corp.    | 1.00  
| Talentum          | 0.89  

This score captures tone instability, which may be relevant for risk-sensitive analysis.

---

## Key Findings

- Fine-tuned FinBERT outperformed both the pretrained transformer and classical baseline in overall accuracy and F1 scores.
- The model generalizes well across sentiment classes and provides stronger performance on subtle polarity shifts.
- Entity-level sentiment aggregation and volatility analysis provided deeper business insight.
- Temporal and linguistic visualization methods enabled more intuitive interpretability of model behavior.

---

## Tools & Stack

- **Modeling**: FinBERT, Hugging Face Transformers, scikit-learn
- **Data**: Hugging Face Datasets (Financial PhraseBank)
- **Visualization**: Matplotlib, Seaborn, WordCloud
