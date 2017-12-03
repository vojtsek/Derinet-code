import numpy as np

def evaluate(gold, hyp):
    tp = fp = tn = fn = 0
    for g, h in zip(gold, hyp):
        if g:
            if h == 1:
                tp += 1
            else:
                fn += 1
        else:
            if h == 0:
                tn += 1
            else:
                fp += 1
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        precision = recall = fscore = 0
    return precision, recall, fscore


def measure_accuracy(gold, hyp):
    acc = 1 - np.sum(np.abs(gold - hyp)) / len(gold)
    triv = 1 - np.sum(gold) / len(gold)
    return triv, acc