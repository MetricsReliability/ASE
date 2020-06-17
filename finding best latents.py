# -*- coding: utf-8 -*-
import math
from multiprocessing import Process, Manager
import pandas as pd
from sklearn.utils import shuffle
import time
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
from sklearn.metrics import roc_auc_score
import csv
import numpy as np
import sys

# my library
from latent_base import AmlmnbBase
from latents import data_k_fold


def calc_perf(true, pred):
    perf_pack = []

    a = classification_report(true, pred, output_dict=True)
    perf_pack.append(round(a['2.0']['precision'], 2))
    perf_pack.append(round(a['2.0']['recall'], 2))
    perf_pack.append(round(a['2.0']['f1-score'], 2))
    ACC = accuracy_score(true, pred)
    perf_pack.append(round(ACC, 2))
    MCC = matthews_corrcoef(true, pred, sample_weight=None)
    perf_pack.append(round(MCC, 2))
    _auc = roc_auc_score(true, pred, average=None)
    perf_pack.append(round(roc_auc_score(true, pred, average=None), 2))

    return perf_pack


# y_test is the vector of class values of the test data
def AUC(c, prob, y_test):
    auc_temp = []
    for j in range(len(c)):
        y_test_binary = []
        for k in range(len(y_test)):
            if y_test[k] == c[j]:
                y_test_binary = y_test_binary + [1]
            else:
                y_test_binary = y_test_binary + [0]
        # tabdil ehtemalat az fazaye logaritm be fazaye vaghei
        real_prob = list(2 ** np.array([prob[i][j] for i in range(len(prob))]))
        if sum(y_test_binary) == len(y_test_binary) or sum(y_test_binary) == 0:
            auc_temp = auc_temp + [1]
        else:
            auc_temp = auc_temp + [roc_auc_score(y_test_binary, real_prob)]
    return sum(auc_temp) / len(auc_temp)


def CLL(c, prob, y_test):
    res1 = []
    for i in range(len(prob)):
        for j in range(len(c)):
            prob[i][j] = 2 ** prob[i][j]
    for i in range(len(y_test)):
        indicator = c.index(y_test[i])
        temp = sum(prob[i])
        res1.append(math.log2(prob[i][indicator] / temp))
    return sum(res1)


def cross_v(learner, x, y, k_fold, iteration):
    acc = []
    auc = []
    cll = []
    for i in range(k_fold):
        x_train, y_train, x_test, y_test = data_k_fold(x, y, k_fold, i)
        # learner.auto_structure(x_train, y_train)
        learner.fit(x_train, y_train, iteration)
        y_pred, prob, c, nominal_pred = learner.predict(x_test)
        out = calc_perf(y_test, nominal_pred)

        acc = acc + [accuracy_score(y_test, nominal_pred)]
        auc = auc + [AUC(c, y_pred, y_test)]
        # auc = auc + [AUC(c, prob, y_test)]
        # cll = cll + [CLL(c, prob, y_test)]
    return acc, auc, cll


def thread_func(learner, x, y, k_fold, iteration, index, acc_res, auc_res, cll_res):
    acc, auc, cll = cross_v(learner, x, y, k_fold, iteration)
    for i in range(k_fold):
        acc_res[index * k_fold + i] = acc[i]
        auc_res[index * k_fold + i] = auc[i]
        # cll_res[index * k_fold + i] = cll[i]


def parallel(data, repeat_cross, k_fold, iteration, learner):
    if __name__ == '__main__':
        manager = Manager()
        accuracy = manager.list([0] * repeat_cross * k_fold)
        AUC = manager.list([0] * repeat_cross * k_fold)
        cll = manager.list([0] * repeat_cross * k_fold)
        threads = []
        for i in range(repeat_cross):
            data = shuffle(data)
            x = list(data[:, :-1])
            y = list(data[:, -1])
            x = [list(x[i]) for i in range(len(x))]
            threads = threads + [
                Process(target=thread_func, args=(learner, x, y, k_fold, iteration, i, accuracy, AUC, cll))]
        for i in range(repeat_cross):
            threads[i].start()
        for i in range(repeat_cross):
            threads[i].join()
        return accuracy, AUC, cll


def save_result_in_csv(ds_name, acc, cll, repeat_cross, k_fold, iteration, AUC, path):
    with open(path, 'w', newline='') as csvfile:
        wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        wr.writerow(['Key_Dataset', 'Iteration', 'repeat_cross', 'fold', 'acc', 'cll', 'AUC'])
        index = 0
        for i in range(repeat_cross):
            for j in range(k_fold):
                wr.writerow([ds_name, iteration, i, j, acc[index], cll[index], AUC[index]])
                index += 1


learner = []
acc = []


def main():
    global learner, acc
    ds_name = "ant"
    data = pd.read_csv('ant0.csv')
    path = "ant.csv"
    data = data.values

    m = np.size(data, 0)

    x = list(data[:, :-1])
    y = list(data[:, -1])
    x = [list(x[i]) for i in range(len(x))]

    t1 = time.time()

    repeat_cross = 10
    eps = sys.float_info.epsilon
    learner = AmlmnbBase(delta=0.01, alpha=0.000001)

    # molecular-biology = {sp= 2, alpha=0.0009, delta=0.001}
    # lymphography = {sp = 20, alpha=0.0009, delta=0.001}

    sp = 2
    # learner.auto_struct(x, y, m, sp, pairwise=True)
    # learner.make_structure(x, y, num_cluster=1, h_states=2)
    k_fold = 10
    iteration = 100
    acc, AUC, cll = parallel(data, repeat_cross, k_fold, iteration, learner)
    # acc,L_h = learner.search_len_h(x, y, iteration, k_fold, [2], delta=0.00002, alpha=0.01)

    t2 = time.time()
    print("accuracy = ", sum(acc) / len(acc))
    print("AUC = ", sum(AUC) / len(AUC))
    # print("CLL = ", sum(cll) / len(cll))
    print("time:", t2 - t1)
    save_result_in_csv(ds_name, acc, cll, repeat_cross, k_fold, iteration, AUC, path)


if __name__ == '__main__':
    main()
