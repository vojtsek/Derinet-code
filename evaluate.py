import sys
import pickle

with open(sys.argv[1], "rb") as f:
    gold = pickle.load(f)
with open(sys.argv[2], "rb") as f:
    hyp = pickle.load(f)

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
print(precision, recall, fscore)
