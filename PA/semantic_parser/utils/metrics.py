from typing import List

def bio_f1(pred_tags: List[List[str]], gold_tags: List[List[str]]) -> float:
    # toy F1: token-level micro F1 for simplicity
    assert len(pred_tags) == len(gold_tags)
    tp = fp = fn = 0
    for p_seq, g_seq in zip(pred_tags, gold_tags):
        for p, g in zip(p_seq, g_seq):
            if p == g and p != "O":
                tp += 1
            elif p != g and p != "O":
                fp += 1
            elif p != g and g != "O":
                fn += 1
    denom = (2*tp + fp + fn)
    return (2*tp / denom) if denom else 1.0

def accuracy(preds: List[int], gts: List[int]) -> float:
    if not gts: return 1.0
    correct = sum(int(p==g) for p,g in zip(preds,gts))
    return correct / len(gts)
