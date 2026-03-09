"""
Credit Card Fraud Detection
Handles extreme class imbalance (0.17% fraud) with SMOTE + ensemble models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, average_precision_score,
                              precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import joblib
import warnings
warnings.filterwarnings('ignore')


def generate_fraud_dataset(n_normal=28000, n_fraud=492, seed=42):
    """Generate PCA-like credit card fraud dataset (mimics the Kaggle dataset structure)."""
    np.random.seed(seed)
    # 28 PCA components V1-V28 + Amount + Time
    n_features = 28

    # Normal transactions
    V_normal = np.random.randn(n_normal, n_features)
    amount_n = np.random.lognormal(3.5, 1.5, n_normal).clip(0, 25691)
    time_n   = np.sort(np.random.uniform(0, 172792, n_normal))

    # Fraud transactions (different distribution)
    V_fraud  = np.random.randn(n_fraud, n_features)
    # Shift certain components to create separation
    V_fraud[:, [0, 2, 4]] -= 2.5  # V1, V3, V5 negative for fraud
    V_fraud[:, [1, 3]]    += 1.5  # V2, V4 positive for fraud
    amount_f = np.random.lognormal(4.0, 1.8, n_fraud).clip(0, 2000)
    time_f   = np.random.uniform(0, 172792, n_fraud)

    V_cols = [f'V{i}' for i in range(1, 29)]
    df_normal = pd.DataFrame(V_normal, columns=V_cols)
    df_normal['Amount'] = amount_n
    df_normal['Time']   = time_n
    df_normal['Class']  = 0

    df_fraud = pd.DataFrame(V_fraud, columns=V_cols)
    df_fraud['Amount'] = amount_f
    df_fraud['Time']   = time_f
    df_fraud['Class']  = 1

    df = pd.concat([df_normal, df_fraud], ignore_index=True).sample(frac=1, random_state=seed)
    return df.reset_index(drop=True)


def preprocess(df):
    """Scale Amount and Time."""
    df = df.copy()
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled']   = scaler.fit_transform(df[['Time']])
    df.drop(['Amount', 'Time'], axis=1, inplace=True)
    return df


def plot_class_distribution(df, save_path='class_distribution.png'):
    plt.figure(figsize=(6, 4))
    counts = df['Class'].value_counts()
    plt.bar(['Normal', 'Fraud'], counts.values, color=['#27AE60', '#E74C3C'])
    for i, v in enumerate(counts.values):
        plt.text(i, v + 50, f'{v:,}\n({v/len(df)*100:.2f}%)', ha='center', fontsize=10)
    plt.title('Transaction Class Distribution')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_fraud_amounts(df, save_path='fraud_amounts.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    df[df['Class']==0]['Amount_scaled'].hist(ax=ax1, bins=50, color='green', alpha=0.7)
    ax1.set_title('Normal Transaction Amounts (scaled)')
    df[df['Class']==1]['Amount_scaled'].hist(ax=ax2, bins=30, color='red', alpha=0.7)
    ax2.set_title('Fraud Transaction Amounts (scaled)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    """Train classifiers optimized for fraud detection."""
    # Apply SMOTETomek for better resampling
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X_train, y_train)
    print(f"After SMOTETomek: {X_res.shape[0]} training samples")
    print(f"  Fraud: {y_res.sum()}, Normal: {(y_res==0).sum()}")

    models = {
        'LogisticRegression': LogisticRegression(C=0.01, class_weight='balanced', max_iter=500),
        'RandomForest':       RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42),
        'XGBoost':            XGBClassifier(n_estimators=100, scale_pos_weight=10,
                                             use_label_encoder=False, eval_metric='aucpr',
                                             max_depth=5, learning_rate=0.1, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_res, y_res)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_prob)
        auc_pr  = average_precision_score(y_test, y_prob)
        # Threshold tuning: maximize F1 on fraud class
        prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
        f1s = 2 * prec * rec / (prec + rec + 1e-8)
        best_thresh = thresholds[np.argmax(f1s[:-1])]
        y_pred = (y_prob >= best_thresh).astype(int)
        results[name] = {'model': model, 'prob': y_prob, 'pred': y_pred,
                          'auc_roc': auc_roc, 'auc_pr': auc_pr, 'thresh': best_thresh}
        print(f"\n{name}: AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}, Threshold={best_thresh:.3f}")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    return results


def plot_precision_recall(results, y_test, save_path='precision_recall.png'):
    plt.figure(figsize=(8, 6))
    for name, r in results.items():
        prec, rec, _ = precision_recall_curve(y_test, r['prob'])
        plt.plot(rec, prec, lw=2, label=f"{name} (AP={r['auc_pr']:.3f})")
    plt.xlabel('Recall (Fraud Detection Rate)')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve — Fraud Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def isolation_forest_analysis(X_train, X_test, y_test):
    """Unsupervised anomaly detection baseline."""
    iso = IsolationForest(n_estimators=100, contamination=0.017, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    scores = -iso.score_samples(X_test)
    auc = roc_auc_score(y_test, scores)
    ap  = average_precision_score(y_test, scores)
    print(f"\nIsolation Forest (unsupervised): AUC-ROC={auc:.4f}, AUC-PR={ap:.4f}")
    return iso


def main():
    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION")
    print("=" * 60)

    df = generate_fraud_dataset()
    print(f"Dataset: {len(df):,} transactions, fraud rate: {df['Class'].mean():.3%}")

    df = preprocess(df)
    plot_class_distribution(df)
    plot_fraud_amounts(df)

    feat_cols = [c for c in df.columns if c != 'Class']
    X = df[feat_cols].values
    y = df['Class'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n--- Unsupervised Baseline ---")
    iso_model = isolation_forest_analysis(X_train, X_test, y_test)

    print("\n--- Supervised Models ---")
    results = train_models(X_train, X_test, y_train, y_test)
    plot_precision_recall(results, y_test)

    best_name = max(results, key=lambda k: results[k]['auc_pr'])
    best      = results[best_name]
    print(f"\nBest model: {best_name} (AUC-PR={best['auc_pr']:.4f})")

    # Financial impact
    n_fraud_detected = (best['pred'] == 1)[y_test == 1].sum()
    n_fraud_total    = y_test.sum()
    avg_fraud_amt    = 150  # assumed average fraud amount in USD
    saved = n_fraud_detected * avg_fraud_amt
    print(f"\nFinancial Impact: Detected {n_fraud_detected}/{n_fraud_total} fraud cases")
    print(f"Estimated savings: ${saved:,.0f}")

    joblib.dump({'model': best['model'], 'features': feat_cols,
                 'threshold': best['thresh']}, 'fraud_model.pkl')
    print("Model saved to fraud_model.pkl")
    print("\n✓ Credit Card Fraud Detection complete!")


if __name__ == '__main__':
    main()
