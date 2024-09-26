from seqeval import metrics
from seqeval.scheme import IOB2


def get_ner_metrics(true_lbs, pred_lbs, mode: str = "strict", scheme=IOB2, detailed: bool = False):
    """
    Get NER metrics including precision, recall and f1

    Parameters
    ----------
    true_lbs: true labels
    pred_lbs: predicted labels
    mode:
    scheme: NER label scheme (IOB-2 as default, [O, B-, I-] )
    detailed: Whether get detailed result report instead of micro-averaged one

    Returns
    -------
    Metrics if not detailed else Dict[str, Metrics]
    """
    if not detailed:
        p = metrics.precision_score(true_lbs, pred_lbs, mode=mode, zero_division=0, scheme=scheme)
        r = metrics.recall_score(true_lbs, pred_lbs, mode=mode, zero_division=0, scheme=scheme)
        f = metrics.f1_score(true_lbs, pred_lbs, mode=mode, zero_division=0, scheme=scheme)
        return {"precision": p, "recall": r, "f1": f}

    else:
        metric_dict = dict()
        report = metrics.classification_report(
            true_lbs, pred_lbs, output_dict=True, mode=mode, zero_division=0, scheme=scheme
        )
        for tp, results in report.items():
            metric_dict[tp] = {
                "precision": results["precision"],
                "recall": results["recall"],
                "f1": results["f1-score"],
            }
        return metric_dict
