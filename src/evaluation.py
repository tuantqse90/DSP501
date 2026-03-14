"""
Evaluation metrics and statistical comparison for Pipeline A vs Pipeline B.

Provides:
- Per-fold and aggregated metrics (accuracy, precision, recall, F1)
- Confusion matrix, ROC curve, AUC
- Statistical tests (paired t-test, Wilcoxon, Cohen's d)
- Results table generation
"""

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import config


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred, average='macro'):
    """Compute classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def compute_per_class_metrics(y_true, y_pred):
    """Compute per-class precision, recall, F1."""
    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES,
                                    output_dict=True, zero_division=0)
    return report


def compute_confusion_matrix(y_true, y_pred, normalize=True):
    """Compute confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    return cm


def compute_roc_curves(y_true, y_scores, n_classes=config.N_CLASSES):
    """
    Compute ROC curves and AUC for each class (one-vs-rest).

    Args:
        y_true: True labels (integer).
        y_scores: Predicted probabilities, shape (n_samples, n_classes).

    Returns:
        Dict with fpr, tpr, auc per class.
    """
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    roc_data = {}

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        roc_data[config.CLASS_NAMES[i]] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

    return roc_data


# ============================================================
# CROSS-VALIDATION AGGREGATION
# ============================================================

def aggregate_fold_results(fold_metrics):
    """
    Aggregate metrics across folds with 95% CI.

    Args:
        fold_metrics: List of dicts from compute_metrics() per fold.

    Returns:
        Dict with mean ± CI for each metric.
    """
    results = {}
    for key in fold_metrics[0]:
        values = np.array([m[key] for m in fold_metrics])
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        ci = 1.96 * std / np.sqrt(len(values))
        results[key] = {
            'mean': float(mean),
            'std': float(std),
            'ci_95': float(ci),
            'values': values.tolist(),
        }
    return results


# ============================================================
# STATISTICAL COMPARISON
# ============================================================

def paired_t_test(scores_a, scores_b):
    """Paired t-test between Pipeline A and B fold scores."""
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return float(t_stat), float(p_value)


def wilcoxon_test(scores_a, scores_b):
    """Wilcoxon signed-rank test (non-parametric alternative)."""
    stat, p_value = stats.wilcoxon(scores_a, scores_b)
    return float(stat), float(p_value)


def cohens_d(scores_a, scores_b):
    """Compute Cohen's d effect size."""
    diff = np.array(scores_a) - np.array(scores_b)
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
    return float(d)


def compare_pipelines(pipeline_a_folds, pipeline_b_folds, metric='accuracy'):
    """
    Full statistical comparison between Pipeline A and B.

    Args:
        pipeline_a_folds: List of metric dicts from Pipeline A (per fold).
        pipeline_b_folds: List of metric dicts from Pipeline B (per fold).
        metric: Which metric to compare.

    Returns:
        Dict with test results.
    """
    scores_a = [m[metric] for m in pipeline_a_folds]
    scores_b = [m[metric] for m in pipeline_b_folds]

    t_stat, p_value_t = paired_t_test(scores_a, scores_b)
    w_stat, p_value_w = wilcoxon_test(scores_a, scores_b)
    effect = cohens_d(scores_a, scores_b)

    return {
        'metric': metric,
        'pipeline_a_mean': float(np.mean(scores_a)),
        'pipeline_b_mean': float(np.mean(scores_b)),
        'difference': float(np.mean(scores_b) - np.mean(scores_a)),
        'paired_t_test': {'t_stat': t_stat, 'p_value': p_value_t},
        'wilcoxon_test': {'w_stat': w_stat, 'p_value': p_value_w},
        'cohens_d': effect,
        'significant_005': p_value_t < 0.05,
    }


# ============================================================
# RESULTS TABLE
# ============================================================

def generate_results_table(all_results):
    """
    Generate a comparison table from all experiment results.

    Args:
        all_results: Dict keyed by (model_name, pipeline_name) → aggregated metrics.

    Returns:
        List of dicts for table rows.
    """
    rows = []
    for (model, pipeline), metrics in all_results.items():
        rows.append({
            'Model': model,
            'Pipeline': pipeline,
            'Accuracy': f"{metrics['accuracy']['mean']*100:.1f} ± {metrics['accuracy']['ci_95']*100:.1f}%",
            'F1 (macro)': f"{metrics['f1']['mean']*100:.1f} ± {metrics['f1']['ci_95']*100:.1f}%",
        })
    return rows
