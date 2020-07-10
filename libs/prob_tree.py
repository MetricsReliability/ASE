import ctypes
import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn import cluster

alpha = 2.220446049250313e-16


def binerize_class(data):
    num_records = data.shape[0]
    for r in range(num_records):
        if data.iloc[r, -1] > 0:
            data.iloc[r, -1] = 2
        else:
            data.iloc[r, -1] = 1
    return data


def code(x_slow, y_slow):
    temp_x = []
    temp_y = []
    compact_attribute = []
    for i in range(len(x_slow[0])):
        temp_x = temp_x + [[]]
        compact_attribute = compact_attribute + [list(set([x_slow[j][i] for j in range(len(x_slow))]))]
        a = [compact_attribute[i].index(x_slow[j][i]) for j in range(len(x_slow))]
        for j in range(len(x_slow)):
            temp_x[i] = temp_x[i] + [a[j]]
    compact_attribute = compact_attribute + [list(set([y_slow[j] for j in range(len(x_slow))]))]
    a = [compact_attribute[len(x_slow[0])].index(y_slow[j]) for j in range(len(x_slow))]
    for j in range(len(x_slow)):
        temp_y = temp_y + [a[j]]

    x = (ctypes.POINTER(ctypes.c_int) * len(x_slow))()
    y = (ctypes.c_int * len(x_slow))()
    for j in range(len(x_slow)):
        x[j] = (ctypes.c_int * len(x_slow[0]))()
        for i in range(len(x_slow[0])):
            x[j][i] = temp_x[i][j]
        y[j] = temp_y[j]
    return x, y, compact_attribute


class BuildStructure:
    def __init__(self, x_slow, y_slow):
        self.I = []
        self.h = []
        self.h_states = []
        x, y, compact_attribute = code(x_slow, y_slow)
        x_dim = len(y_slow)
        y_dim = len(x_slow[0])
        mutual = (ctypes.POINTER(ctypes.c_double) * y_dim)()
        for i in range(y_dim):
            mutual[i] = (ctypes.c_double * y_dim)()
        x_dim = ctypes.c_int(x_dim)
        y_dim = ctypes.c_int(y_dim)

        imc = ctypes.CDLL('E:\\apply\\york\\project\\source\\find_mutual_conditional.dll')
        imc.information_mutual_conditional_all(x, y, x_dim, y_dim, mutual)

        y_dim = y_dim.value
        for i in range(y_dim):
            self.I = self.I + [[]]
            for j in range(y_dim):
                self.I[i] = self.I[i] + [mutual[i][j]]

    def structure(self, n_h):
        y_dim = len(self.I)
        cluster = []
        for i in range(y_dim):
            # initial cluster size which is equal to the number of attributes in the dataset
            cluster = cluster + [set([i])]
        # n_h is the number of clusters determined by the user
        while len(cluster) > n_h:
            distance = []
            for i in range(len(cluster)):
                distance = distance + [[]]
                for _ in range(len(cluster)):
                    distance[i] = distance[i] + [0]
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    distance[i][j] = 0
                    for k in cluster[i]:
                        for l in cluster[j]:
                            distance[i][j] += self.I[k][l]
                    distance[i][j] /= (len(cluster[i]) * len(cluster[j]))
                    distance[j][i] = distance[i][j]
            index_i = 0
            index_j = 1
            maximum = distance[0][1]
            for i in range(len(distance)):
                for j in range(i + 1, len(distance)):
                    if maximum < distance[i][j]:
                        maximum = distance[i][j]
                        index_i = i
                        index_j = j
            cluster[index_i] = cluster[index_i] | cluster[index_j]
            del cluster[index_j]

        tree = []
        for i in range(len(cluster)):
            if len(cluster[i]) > 1:
                tree.append(list(cluster[i]))

        return tree


def make_prob(whole_data, tree_indicator):
    num_tree = 2
    # tree_indicator = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20]]
    tree_params = []
    uc = np.unique(whole_data.iloc[:, -1])

    for i in range(len(tree_indicator)):
        tree_params = tree_params + [[]]
        current_tree = tree_indicator[i]
        # unique_holder = unique_holder + [[]]
        # size_holder = size_holder + [[]]
        for c in range(len(uc)):
            tree_params[i] = tree_params[i] + [[]]
            for j in range(len(current_tree)):
                aj = current_tree[j]
                uai = np.unique(whole_data.iloc[:, aj])
                tree_params[i][c] = tree_params[i][c] + [[]]
                # unique_holder[i] = unique_holder[i] + [uai]
                # size_holder[i] = size_holder[i] + [uai]
                for k in range(np.size(uai, 0)):
                    tree_params[i][c][j] = tree_params[i][c][j] + [[]]

    return tree_params, tree_indicator


def product(ar_list):
    if not ar_list:
        yield ()
    else:
        for a in ar_list[0]:
            for prod in product(ar_list[1:]):
                yield (a,) + prod


def fit(tree_params, tree_indicator, data):
    n, N = data.shape
    uc = np.unique(data.iloc[:, -1])
    pclass = np.zeros((1, len(uc)))
    for c in range(len(uc)):
        pclass[0, c] = len(data[data.iloc[:, -1] == uc[c]]) / n

    unique_holder = []
    size_holder = []
    for i in range(len(tree_params)):
        unique_holder = unique_holder + [[]]
        size_holder = size_holder + [[]]
        current_cluster = tree_indicator[i]
        for j in range(len(current_cluster)):
            aj = current_cluster[j]
            uaj = list(np.unique(data.iloc[:, aj]))
            unique_holder[i] = unique_holder[i] + [uaj]
            size_holder[i] = size_holder[i] + [len(uaj)]
        # unique_holder[i] = list(product(unique_holder[i]))

    for i in range(len(tree_params)):
        current_cluster = tree_indicator[i]
        for c in range(len(uc)):
            # for u in unique_holder[i]:
            # for k, val in enumerate(u):
            for j in range(len(current_cluster)):
                aj = current_cluster[j]
                # for k in range(np.size(tree_params[i][c][j], 0)):
                for k in range(size_holder[i][j]):
                    idx = unique_holder[i][j]
                    tree_params[i][c][j][k] = (
                            len(np.where((data.iloc[:, aj] == idx[k]) & (data.iloc[:, -1] == uc[c]))[0]) / len(
                        np.where((data.iloc[:, -1] == uc[c]))[0]))
                    if tree_params[i][c][j][k] == 0:
                        tree_params[i][c][j][k] = 0.01
                    tree_params[i][c][j][k] = np.log2(tree_params[i][c][j][k])
    return tree_params, pclass, unique_holder


def predict(tree_params, pclass, tree_indicator, test_data, unique_holder):
    output = []
    prob = []
    for c in range(np.size(pclass, 1)):
        ti = 0
        prob = prob + [[]]
        for i in range(np.size(tree_params, 0)):
            current_cluster = tree_indicator[i]
            temp = 0
            for j in range(np.size(current_cluster, 0)):
                aj = current_cluster[j]
                idx = test_data.iloc[aj]
                if idx in unique_holder[i][j]:
                    index = unique_holder[i][j].index(idx)
                    temp += tree_params[i][c][j][index]
                else:
                    temp += np.log2(1 / np.size(pclass, 1))
            ti += temp
        # ti += pclass[0, c]
        prob[c] = ti
        prob[c] = 2 ** prob[c]
    output.append(prob)
    return np.argmax(prob)


def identify_subtree(data):
    X = data.iloc[:, 0:-1].values
    y = data.iloc[:, -1].values
    struct = BuildStructure(X, y)
    n, N = data.shape
    if N <= 5:
        n_cluster = 2
    else:
        n_cluster = round(N / 2)
    sub_trees = struct.structure(n_cluster)
    return sub_trees


def main():
    data = pd.read_csv("E:\\apply\\york\\project\\source\\datasets\\credit-g.csv")
    #train_data = pd.read_csv("xerces-1.2.csv")
    #test_data = pd.read_csv("xerces-1.3.csv")

    #train_data = binerize_class(train_data)
    #test_data = binerize_class(test_data)

    train_data, test_data = train_test_split(data, shuffle=True, test_size=0.33, random_state=42)

    clust = identify_subtree(train_data)

    tree_params, tree_indicator = make_prob(train_data, clust)
    tree_params, pclass, unique_holder = fit(tree_params, clust, train_data)
    pred = []
    for i in range(len(test_data)):
        pred.append(predict(tree_params, pclass, clust, test_data.iloc[i, :], unique_holder))

    print("CITree Classification Accuracy:", accuracy_score(test_data.iloc[:, -1], pred))
    print(classification_report(test_data.iloc[:, -1], pred))
    # print("AUC:", roc_auc_score(test_data.iloc[:, -1], pred, average=None))


if __name__ == '__main__':
    main()
