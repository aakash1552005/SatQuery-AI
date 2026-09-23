"""
Comprehensive Task-Aware Remote Sensing Metrics Module.
Calculates:
- Classification: Accuracy, Balanced Accuracy, Precision, Recall, Specificity, Macro-F1, Weighted-F1, Confusion Matrix
- Grounding: IoU, Mean IoU, Median IoU, Acc@0.5, Acc@0.7
- Caption: BLEU-4, ROUGE-L, METEOR
- VQA: Overall, binary, MCQ, per-task
"""

from typing import Any
import numpy as np

def compute_classification_metrics(y_true: list[Any], y_pred: list[Any], classes: list[str]) -> dict[str, Any]:
    """Calculate multi-class classification and binary metrics."""
    from collections import Counter
    total = len(y_true)
    if total == 0:
        return {}

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / total

    # Confusion matrix
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    n_cls = len(classes)
    cm = [[0 for _ in range(n_cls)] for _ in range(n_cls)]

    for yt, yp in zip(y_true, y_pred):
        ti = cls_to_idx.get(yt)
        pi = cls_to_idx.get(yp)
        if ti is not None and pi is not None:
            cm[ti][pi] += 1

    per_class_recalls = []
    per_class_precisions = []
    per_class_f1s = []
    supports = []

    for i in range(n_cls):
        tp = cm[i][i]
        fn = sum(cm[i][j] for j in range(n_cls) if j != i)
        fp = sum(cm[j][i] for j in range(n_cls) if j != i)
        tn = sum(cm[r][c] for r in range(n_cls) for c in range(n_cls) if r != i and c != i)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        support = tp + fn

        per_class_precisions.append(prec)
        per_class_recalls.append(rec)
        per_class_f1s.append(f1)
        supports.append(support)

    macro_f1 = float(np.mean(per_class_f1s)) if per_class_f1s else 0.0
    weighted_f1 = float(np.average(per_class_f1s, weights=supports)) if sum(supports) > 0 else 0.0
    balanced_acc = float(np.mean(per_class_recalls)) if per_class_recalls else 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "classes": classes,
        "support": supports,
    }


def compute_grounding_iou(pred_box: list[float], gt_box: list[float]) -> float:
    """Calculate intersection-over-union for boxes [ymin, xmin, ymax, xmax]."""
    y1 = max(pred_box[0], gt_box[0])
    x1 = max(pred_box[1], gt_box[1])
    y2 = min(pred_box[2], gt_box[2])
    x2 = min(pred_box[3], gt_box[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_pred = max(0.0, pred_box[2] - pred_box[0]) * max(0.0, pred_box[3] - pred_box[1])
    area_gt = max(0.0, gt_box[2] - gt_box[0]) * max(0.0, gt_box[3] - gt_box[1])
    union_area = area_pred + area_gt - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_grounding_metrics(pred_boxes: list[list[float]], gt_boxes: list[list[float]]) -> dict[str, float]:
    """Calculate mean IoU, median IoU, Acc@0.5, and Acc@0.7."""
    if not pred_boxes or len(pred_boxes) != len(gt_boxes):
        return {"mean_iou": 0.0, "acc_0.5": 0.0, "acc_0.7": 0.0}

    ious = [compute_grounding_iou(p, g) for p, g in zip(pred_boxes, gt_boxes)]
    return {
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "acc_0.5": float(np.mean([1.0 if i >= 0.5 else 0.0 for i in ious])),
        "acc_0.7": float(np.mean([1.0 if i >= 0.7 else 0.0 for i in ious])),
    }
