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

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2 * precision * recall / (precision + recall)
    return precision, recall, fscore


# def measure_accuracy():