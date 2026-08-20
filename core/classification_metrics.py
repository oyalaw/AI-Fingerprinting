"""Real classification-metrics computation for FL per-round evaluate()
phases -- hand-rolled from confusion counts rather than adding
scikit-learn as a new dependency for four numbers.

Macro-averaged (each class weighted equally, not by support): the
datasets FL/distributed training uses (core/training_data.py's
stride-partitioning across clients) are already close to class-balanced
by construction, and macro is the standard choice for per-round FL
evaluation in the traffic-fingerprinting literature this project's own
feature set is otherwise modeled on.
"""


def compute_classification_metrics(predictions, labels, num_classes):
    """predictions/labels: equal-length 1-D LongTensors of predicted and
    true class indices. Returns {accuracy, precision_score, recall,
    f1_score} -- "precision_score", not "precision": every FL adapter's
    event log already stamps a "precision" field on every record via
    core/experiment.py's _common_event_fields() (fp32/fp16/etc, numeric
    precision), a real, confirmed key collision with the classification
    metric of the same name if this used that name too. A class with zero
    predicted-or-true examples contributes 0.0 to its own precision/
    recall/F1 (not NaN) -- a small per-round FL client partition often
    doesn't include every class."""
    if len(predictions) == 0:
        return {"accuracy": 0.0, "precision_score": 0.0, "recall": 0.0, "f1_score": 0.0}

    accuracy = int((predictions == labels).sum()) / len(labels)

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = int(((predictions == c) & (labels == c)).sum())
        fp = int(((predictions == c) & (labels != c)).sum())
        fn = int(((predictions != c) & (labels == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "precision_score": sum(precisions) / num_classes,
        "recall": sum(recalls) / num_classes,
        "f1_score": sum(f1s) / num_classes,
    }
