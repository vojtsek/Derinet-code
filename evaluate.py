import glob
import sys

hyp = []
with open(sys.argv[2], "r") as f:
    for c in f.read():
        hyp.append(int(c))
def evaluate_file(fn):
    gold = []
    with open(sys.argv[1], "r") as f:
        for c in f.read():
            gold.append(int(c))

    tp = fp = tn = fn = 0
    correct = 0
    negatives = 0
    tp2 = fp2 = tn2 = fn2 = 0
    for g, h in zip(gold, hyp):
        if g:
            fn2 += 1
            if h == 1:
                tp += 1
                correct += 1
            else:
                fn += 1
        else:
            negatives += 1
            tn2 += 1
            if h == 0:
                tn += 1
                correct += 1
            else:
                fp += 1

    alll = len(gold)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2 * precision * recall / (precision + recall)
    print("Precision: {}, recall: {}, fscore: {}, accuracy: {}, triv. accuracy: {}".format(precision, recall, fscore, (correct / alll), (negatives / alll)))
    return fscore

results_train = []
results_valid = []
results_test = []
for fn in glob.glob("epoch_layer-size-{}*".format(sys.argv[1])):
    fscore = evaluate_file(fn)
    if fn.endswith("train.pred"):
        results_train.append(fscore)
    elif fn.endswith("valid.pred"):
        results_valid.append(fscore)
    elif fn.endswith("test.pred"):
        results_valid.append(fscore)
