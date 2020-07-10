import itertools

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

alpha = 2.220446049250313e-16

def make_prob(data, whole_data):
    num_tree = 2
    tree_indicator = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20]]
    tree_params = []
    uc = np.unique(whole_data.iloc[:, -1])

    for i in range(num_tree):
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


def fit(tree_params, tree_indicator, data, whole_data):
    n, N = data.shape
    uc = np.unique(whole_data.iloc[:, -1])
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
            uaj = list(np.unique(whole_data.iloc[:, aj]))
            unique_holder[i] = unique_holder[i] + [uaj]
            size_holder[i] = size_holder[i] + [uaj]
        #unique_holder[i] = list(product(unique_holder[i]))

    for i in range(len(tree_params)):
        current_cluster = tree_indicator[i]
        for c in range(len(uc)):
            # for u in unique_holder[i]:
            # for k, val in enumerate(u):
            for j in range(len(current_cluster)):
                aj = current_cluster[j]
                for k in range(np.size(tree_params[i][c][j], 0)):
                    idx = unique_holder[i][j]
                    tree_params[i][c][j][k] = (
                            len(np.where((data.iloc[:, aj] == idx[k]) & (data.iloc[:, -1] == uc[c]))[0]) / len(
                        np.where((data.iloc[:, -1] == uc[c]))[0]))
                    if tree_params[i][c][j][k] == 0:
                        tree_params[i][c][j][k] = alpha
    return tree_params, pclass, unique_holder


def predict(tree_params, pclass, tree_indicator, test_data, unique_holder):
    output = []
    prob = []
    for c in range(np.size(pclass, 1)):
        ti = 1
        prob = prob + [[]]
        for i in range(np.size(tree_params, 0)):
            current_cluster = tree_indicator[i]
            temp = 1
            for j in range(np.size(current_cluster, 0)):
                aj = current_cluster[j]
                idx = test_data.iloc[aj]
                if idx in unique_holder[i][j]:
                    index = unique_holder[i][j].index(idx)
                    temp *= tree_params[i][c][j][index]
                else:
                    temp *= np.log2(1 / len(np.size(pclass, 1)))
            ti *= temp
        prob[c] = ti
    output.append(prob)

    # pred_y = np.zeros((2, len(tree_indicator)))
    # for i in range(len(tree_indicator)):
    #     pred_y[0, i] = max(output[i])
    #     pred_y[1, i] = np.argmax(output[i])
    return np.argmax(prob)


def main():
    data = pd.read_csv("E:\\apply\\york\\project\\source\\datasets\\credit-g.csv")

    train_data, test_data = train_test_split(data, shuffle=True, test_size=0.33, random_state=42)

    tree_params, tree_indicator = make_prob(train_data, data)
    tree_params, pclass, unique_holder = fit(tree_params, tree_indicator, train_data, data)
    pred = []
    for i in range(len(test_data)):
        pred.append(predict(tree_params, pclass, tree_indicator, test_data.iloc[i, :], unique_holder))

    print("CITree Classification Accuracy:", accuracy_score(test_data.iloc[:, -1], pred))


if __name__ == '__main__':
    main()
