"""
Shared utilities for model testing: confusion matrix printer,
per-class stats, colour helpers.
"""

import numpy as np
from collections import defaultdict


def confusion_matrix(y_true, y_pred, class_names):
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    idx = {c: i for i, c in enumerate(class_names)}
    for t, p in zip(y_true, y_pred):
        cm[idx[t]][idx[p]] += 1
    return cm


def print_confusion_matrix(cm, class_names):
    col_w = 10
    header = f"{'':12}" + "".join(f"{c:>{col_w}}" for c in class_names)
    print(header)
    for i, row_label in enumerate(class_names):
        row = f"{row_label:12}" + "".join(f"{cm[i][j]:>{col_w}}" for j in range(len(class_names)))
        print(row)


def per_class_stats(cm, class_names):
    stats = {}
    for i, cls in enumerate(class_names):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support   = cm[i, :].sum()
        stats[cls] = dict(precision=precision, recall=recall, f1=f1, support=int(support), tp=int(tp))
    return stats


def print_per_class_stats(stats):
    print(f"\n{'Class':16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)
    for cls, s in stats.items():
        print(f"{cls:16} {s['precision']:>10.3f} {s['recall']:>10.3f} {s['f1']:>10.3f} {s['support']:>10}")


def macro_avg(stats):
    prec = np.mean([s['precision'] for s in stats.values()])
    rec  = np.mean([s['recall']    for s in stats.values()])
    f1   = np.mean([s['f1']        for s in stats.values()])
    return prec, rec, f1


def print_summary_box(title, lines):
    w = max(len(l) for l in lines) + 4
    border = "═" * w
    print(f"\n╔{border}╗")
    print(f"║  {title.center(w - 2)}  ║")
    print(f"╠{border}╣")
    for line in lines:
        print(f"║  {line:<{w - 2}}║")
    print(f"╚{border}╝")
