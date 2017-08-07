import csv
import numpy as np
import editdistance
import pickle
import io
# make pairs from windows of size 10
# -> l1,l2,deriv?

def create_features(dato):
    pass

def get_halves(w1, w2):
    i = 0
    for i, c in enumerate(zip(w1, w2)):
        if c[0] != c[1]:
            break
    suff1 = w1[i:]
    suff2 = w2[i:]
    if suff1 > suff2:
        tmp = suff1
        suff1 = suff2
        suff2 = tmp
    return w1[:i], (suff1, suff2)


class Dataset:
    def __init__(self, fn, test_size=0.2, valid_size=0.1):
        suffix_pairs = {}
        SUFFIX_COUNT = 100
        # read and analyze the data
        with io.open(fn, "r", encoding="utf-8") as dest_f:
            data_iter = csv.reader(dest_f,
                                   delimiter='\t',
                                   quotechar='"')
            data = [data for data in data_iter]
        data = np.asarray(data)
        pairs = []
        for n, line in enumerate(data):
            for i in range(-5, 6):
                if i == 0:
                    continue
                idx = n + i
                if idx < 0 or idx >= len(data):
                    continue
                first = line
                second = data[idx, :]
                edge = line[0] == data[idx, -1]
                _, (suff1, suff2) = get_halves(first[1], second[1])
                pairs.append((first, second, edge))
                try:
                    suffix_pairs[(suff1, suff2)] += 1
                except:
                    suffix_pairs[(suff1, suff2)] = 1

        d = sorted([(y, x) for x, y in suffix_pairs.items()], reverse=True)
        suffix_features = {y[1]: x for x, y in enumerate(d[:SUFFIX_COUNT])}

        # construct feature vector
        X = []
        y = []
        for datapoint in pairs:
            fv = np.zeros((SUFFIX_COUNT + 1 + 6))
            first, second, edge = datapoint
            common_prefix, (suff1, suff2) = get_halves(first[1], second[1])
            try:
                idx = suffix_features[(suff1, suff2)]
                fv[idx] = 1
            except:
                fv[SUFFIX_COUNT] = 1  # other
            fv[SUFFIX_COUNT + 1] = len(common_prefix)
            fv[SUFFIX_COUNT + 2] = editdistance.eval(first[1], second[1])
            fv[SUFFIX_COUNT + 3] = str.isupper(first[1][0])
            fv[SUFFIX_COUNT + 4] = str.isupper(second[1][0])
            fv[SUFFIX_COUNT + 5] = ord(first[3])
            fv[SUFFIX_COUNT + 6] = ord(second[3])
            X.append(fv)
            y.append(edge)

        X = np.array(X)
        y = np.array(y)
        dataset = (X, y)
        test_size = int(len(X) * test_size)
        valid_size = int(len(X) * valid_size)
        self.train_X, self.train_y = X[(test_size+valid_size):], y[(test_size+valid_size):]
        self.valid_X, self.valid_y = X[test_size:(test_size + valid_size)], y[test_size:(test_size + valid_size)]
        self.test_X, self.test_y = X[:test_size], y[:test_size]
        self.current_idx = 0

    def has_next(self):
        return self.current_idx < len(self.train_X)

    def get_test(self):
        return self.test_X, self.test_y

    def get_train(self):
        return self.train_X, self.train_y

    def get_valid(self):
        return self.valid_X, self.valid_y

    def next_batch(self, bs=10):
        batch = self.train_X[self.current_idx:(self.current_idx + bs)], self.train_y[self.current_idx:(self.current_idx + bs)]
        self.current_idx += bs
        return batch

    def reset(self):
        self.current_idx = 0


"""
 featury    - spolecna zakonceni - odseknout prefix, seradit podle poctu vyskytuderivaci
            - delka prefixu
            - Lev. Distance
            - per Lemma slovni druh,
"""
