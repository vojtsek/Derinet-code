import csv
import numpy as np
import editdistance
import pickle
import io
# make pairs from windows of size 10
# -> l1,l2,deriv?

import unicodedata

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii



def get_all_bigrams(word):
    bigrams = set()
    for w1,w2 in zip(word, word[1:]):
        bigr = w1+w2
        unaccented = remove_accents(bigr)
        bigrams.add(unaccented)
    return bigrams

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


class DataSubset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def get_data(self):
        return self.X, self.y

    def get_data_as_chars(self):
        pass

class Dataset:
    def __init__(self, fn, test_size=0.2, as_chars=False):
        suffix_pairs = {}
        SUFFIX_COUNT = 100
        self.dst_from_parents = []
        self.chars2ints = {}
        self.sentence_lens = []
        self.bigrams = set()
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
                # for bigr in get_all_bigrams(first[1]):
                #     self.bigrams.add(bigr)
                edge = line[0] == data[idx, -1]
                if edge:
                    self.dst_from_parents.append(np.abs(int(line[0]) - int(data[idx, 0])))
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
        max_sentence_length = 0
        for datapoint in pairs:
            first, second, edge = datapoint
            if not as_chars:
                fv = np.zeros((SUFFIX_COUNT + 1 + 6))
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
            else:
                sentence_length = len(first[1]) + len(second[1]) + 1
                max_sentence_length = max(max_sentence_length, sentence_length)
                self.sentence_lens.append(sentence_length)
                X.append(self.embed(first[1], second[1]))
            y.append(edge)
        if not as_chars:
            X_tmp = np.array(X)
        else:
            X_tmp = np.zeros((len(X), max_sentence_length))
            for i, rec in enumerate(X):
                X_tmp[i,:len(rec)] = rec
        y = np.array(y)
        X = X_tmp
        self.dataset = (X, y)
        # with open('dataset-subset.dump', 'wb') as f:
        #     pickle.dump(dataset, f)
        test_size = int(len(X) * test_size)
        self.sentence_lens = np.array(self.sentence_lens)
        self.train_X, self.train_y, self.lens_train = X[test_size:], y[test_size:], self.sentence_lens[test_size:]
        self.perm = np.random.permutation(len(self.train_X))
        self.test_X, self.test_y, self.lens_test = X[:test_size], y[:test_size], self.sentence_lens[:test_size]
        self.current_idx = 0

    def embed(self, v1, v2):
        result = []
        for v in v1:
            if v not in self.chars2ints:
                # 0, 1 reserved for separator and UNK
                self.chars2ints[v] = len(self.chars2ints) + 2
            result.append(self.chars2ints[v])
        result.append(0)
        for v in v2:
            if v not in self.chars2ints:
                # 0, 1 reserved for separator and UNK
                self.chars2ints[v] = len(self.chars2ints) + 2
            result.append(self.chars2ints[v])
        return result

    def has_next(self):
        if self.current_idx + 10 >= len(self.train_X):
            return False
        return True

    def get_test(self):
        return self.test_X, self.test_y, self.lens_test

    def get_train(self):
        return self.train_X, self.train_y, self.lens_train

    def next_batch(self, bs=10):
        batch = self.train_X[self.perm[self.current_idx:(self.current_idx + bs)], :],\
                self.train_y[self.perm[self.current_idx:(self.current_idx + bs)]],\
                self.lens_train[self.perm[self.current_idx:(self.current_idx + bs)]]
        self.current_idx += bs
        return batch

    def reset(self):
        self.current_idx = 0
        self.perm = np.random.permutation(len(self.train_X))

if __name__ == '__main__':
    d = Dataset('data/derismall.tsv', as_chars=True)
    d.next_batch()
    # d.observe()


"""
 featury    - spolecna zakonceni - odseknout prefix, seradit podle poctu vyskytuderivaci
            - delka prefixu
            - Lev. Distance
            - per Lemma slovni druh,
"""
