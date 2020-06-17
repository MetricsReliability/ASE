# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:19:13 2019

@author: ALIREZA HEDIEHLOO
mail: alirezahediehloo@gmail.com
"""

import numpy as np
import math
from sklearn.metrics import accuracy_score

# latent library
from latent_base import AmlmnbBase
from make_structure import MakeStructure


def data_k_fold(x, y, k_fold, i):
    x_dim = len(y)
    if i == 0:
        start = int(x_dim/k_fold)
        x_train = x[start:]
        y_train = y[start:]
        x_test = x[:start]
        y_test = y[:start]
    elif i == k_fold-1:
        end = int(i*x_dim/k_fold)
        x_train = x[:end]
        y_train = y[:end]
        x_test = x[end:]
        y_test = y[end:]
    else:
        end1 = int(i*x_dim/k_fold)
        x_train = x[:end1]
        y_train = y[:end1]
        start2 = int((i+1)*x_dim/k_fold)
        x_train = x_train + x[start2:]
        y_train = y_train + y[start2:]
        x_test = x[end1:start2]
        y_test = y[end1:start2]
    return x_train, y_train, x_test, y_test


def find_indexes(show_l, length):
    indexes = []
    for i in range(len(length)-1, -1, -1):
        a = int(show_l % length[i])
        indexes = [a] + indexes
        show_l -= a
        show_l /= length[i]
    return indexes


def fold_accuracy(probabilities, indexes, c, pc, y_test):
    p_x_c = np.zeros((len(y_test), len(c)))
    p_x_c += np.array([pc]*len(y_test))
    for i in range(len(probabilities)):
        p_x_c += probabilities[i][indexes[i]]
    y_pred = []
    p_x_c = [list(p_x_c[i]) for i in range(len(p_x_c))]
    for i in range(len(y_test)):
        y_pred = y_pred + [c[p_x_c[i].index(max(p_x_c[i]))]]
    acc = accuracy_score(y_test, y_pred)
    return acc


def accuracy(probabilities, indexes, c, pc, y_test):
    acc = []
    for i in range(len(probabilities)):
        acc = acc + [fold_accuracy(probabilities[i], indexes, c[i], pc[i], y_test[i])]
    a = sum(acc)/len(acc)
    return a


class AMLMNB:
    def __init__(self, delta=1e-10, alpha=0, mode_h='individual'):
        self.structure = []
        self.base_learner = []
        self.delta = delta
        self.alpha = alpha
        self.mode_h = mode_h  # 'joint' or 'individual'
        self.convergence = []
        
    def make_structure(self, x, y, num_cluster, h_states):
        ms = MakeStructure(x, y)
        self.structure = ms.structure(num_cluster, h_states)

    def fit(self, x, y, iteration=100):
        self.base_learner = []
        for i in range(len(self.structure.h)):
            self.base_learner = self.base_learner + [AmlmnbBase(
                        [[k for k in range(len(self.structure.h[i]))]], [self.structure.h_states[i]],
                        delta=self.delta, alpha=self.alpha, mode_h=self.mode_h)]
        
        for i in range(len(self.structure.h)):
            x_one_h = []
            for j in range(len(x)):
                x_one_h = x_one_h + [[x[j][k] for k in self.structure.h[i]]]
            self.base_learner[i].fit(x_one_h, y, iteration)
        
        self.convergence = []
        for j in range(iteration):
            self.convergence = self.convergence + [0]
            for i in range(len(self.structure.h)):
                self.convergence[j] += self.base_learner[i].convergence[j]
            if self.convergence[j] != 0:
                self.convergence[j] = math.log2(self.convergence[j])
            else:
                self.convergence[j] = -np.inf

    """------------------------------"""
    def continue_(self, iteration=100):
        for i in range(len(self.structure.h)):
            self.base_learner[i].continue_(iteration)
        
        self.convergence = []
        for j in range(len(self.base_learner[0].convergence)):
            self.convergence = self.convergence + [0]
            for i in range(len(self.structure.h)):
                self.convergence[j] += self.base_learner[i].convergence[j]
            if self.convergence[j] != 0:
                self.convergence[j] = math.log2(self.convergence[j])
            else:
                self.convergence[j] = -np.inf

    def predict(self, x):
        c = self.base_learner[0].compact_attribute[len(self.base_learner[0].compact_attribute)-1]
        p_x_c = np.zeros((len(x), len(c)))
        p_x_c += np.array([self.base_learner[0].pc]*len(x))
        for i in range(len(self.base_learner)):
            x_one_h = []
            for j in range(len(x)):
                x_one_h = x_one_h + [[x[j][k] for k in self.structure.h[i]]]
            p_x_c += np.array(self.base_learner[i].predict(x_one_h))
        y_pred = []
        p_x_c = [list(p_x_c[i]) for i in range(len(p_x_c))]
        for i in range(len(x)):
            y_pred = y_pred + [c[p_x_c[i].index(max(p_x_c[i]))]]
        return y_pred, p_x_c, c

    def find_prob(self, x_train, y_train, x_test, h_states, iteration=100, delta=1e-10, alpha=0):
        base_learner = []
        for i in range(len(self.structure.h)):
            base_learner = base_learner + [[]]
            for j in range(len(h_states)):
                base_learner[i] = base_learner[i] + [AmlmnbBase(
                        [[k for k in range(len(self.structure.h[i]))]], [h_states[j]],
                        delta, alpha,
                        mode_h='individual')]
                if len(self.structure.h[i]) == 1:
                    break
        
        for i in range(len(self.structure.h)):
            x_one_h = []
            for k in range(len(x_train)):
                x_one_h = x_one_h + [[x_train[k][l] for l in self.structure.h[i]]]
            for j in range(len(base_learner[i])):
                base_learner[i][j].fit(x_one_h, y_train, iteration)
        
        c = base_learner[0][0].compact_attribute[-1]
        pc = base_learner[0][0].pc
        
        prob = []
        for i in range(len(self.structure.h)):
            prob = prob + [[]]
            x_one_h = []
            for j in range(len(x_test)):
                x_one_h = x_one_h + [[x_test[j][k] for k in self.structure.h[i]]]
            for j in range(len(base_learner[i])):
                prob[i] = prob[i] + [np.array(base_learner[i][j].predict(x_one_h))]
        
        return prob, c, pc

    def search_len_h(self, x, y, iteration, k_fold, h_states, delta=1e-10, alpha=0):
        probabilities = []
        c = []
        pc = []
        y_test = []
        for i in range(k_fold):
            y_test = y_test + [[]]
            x_train, y_train, x_test, y_test[i] = data_k_fold(x, y, k_fold, i)
            
            probabilities = probabilities + [[]]
            c = c + [[]]
            pc = pc + [[]]
            probabilities[i], c[i], pc[i] = \
                self.find_prob(x_train, y_train, x_test, h_states, iteration, delta, alpha)
        
        length = []
        end_length = 1
        for i in range(len(self.structure.h)):
            if len(self.structure.h[i]) > 1:
                length = length + [len(h_states)]
            else:
                length = length + [1]
            end_length *= length[i]

        acc = []
        num_h_state = []
        for show_L in range(end_length):
            indexes = find_indexes(show_L, length)
            num_h_state = num_h_state + [[h_states[indexes[i]] for i in range(len(indexes))]]
            acc = acc + [accuracy(probabilities, indexes, c, pc, y_test)]
        return acc, num_h_state
