# Credit Card Fraud Detection

Detects fraudulent transactions from a highly imbalanced dataset (0.17% fraud rate) using supervised and unsupervised methods.

## Features
- Synthetic PCA-based credit card transaction dataset (mimics Kaggle structure)
- SMOTETomek resampling for extreme class imbalance
- Threshold tuning via precision-recall curve
- Isolation Forest for unsupervised anomaly baseline
- Financial impact estimation

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Output
- `class_distribution.png` — fraud vs normal bar chart
- `fraud_amounts.png` — amount distributions
- `precision_recall.png` — PR curves for all models
- `fraud_model.pkl` — saved model + optimal threshold

## Key Metric
We optimize AUC-PR (Average Precision) rather than accuracy, since the dataset is highly imbalanced.
