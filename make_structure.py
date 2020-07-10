# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:20:47 2019

@author: ALIREZA HEDIEHLOO
mail: alirezahediehloo@gmail.com
"""

import numpy as np
import ctypes


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

    x = (ctypes.POINTER(ctypes.c_int)*len(x_slow))()
    y = (ctypes.c_int*len(x_slow))()
    for j in range(len(x_slow)):
        x[j] = (ctypes.c_int*len(x_slow[0]))()
        for i in range(len(x_slow[0])):
            x[j][i] = temp_x[i][j]
        y[j] = temp_y[j]
    return x, y, compact_attribute


def is2vct_equal(a, b):
    for i in range(len(a)):
        if a[i] != b[i]:
            return 0
    return 1


class MakeStructure:
    def __init__(self, x_slow, y_slow):
        self.I = []
        self.h = []
        self.h_states = []
        x, y, compact_attribute = code(x_slow, y_slow)
        x_dim = len(y_slow)
        y_dim = len(x_slow[0])
        mutual = (ctypes.POINTER(ctypes.c_double)*y_dim)()
        for i in range(y_dim):
            mutual[i] = (ctypes.c_double*y_dim)()
        x_dim = ctypes.c_int(x_dim)
        y_dim = ctypes.c_int(y_dim)
        
        imc = ctypes.CDLL('E:\\apply\\york\\project\\source\\find_mutual_conditional.dll')
        imc.information_mutual_conditional_all(x, y, x_dim, y_dim, mutual)
        
        y_dim = y_dim.value
        for i in range(y_dim):
            self.I = self.I + [[]]
            for j in range(y_dim):
                self.I[i] = self.I[i] + [mutual[i][j]]

    def structure(self, n_h, h_states):
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
                for j in range(i+1, len(cluster)):
                    distance[i][j] = 0
                    for k in cluster[i]:
                        for l in cluster[j]:
                            distance[i][j] += self.I[k][l]
                    distance[i][j] /= (len(cluster[i])*len(cluster[j]))
                    distance[j][i] = distance[i][j]
            index_i = 0
            index_j = 1
            maximum = distance[0][1]
            for i in range(len(distance)):
                for j in range(i+1, len(distance)):
                    if maximum < distance[i][j]:
                        maximum = distance[i][j]
                        index_i = i
                        index_j = j
            cluster[index_i] = cluster[index_i] | cluster[index_j]
            del cluster[index_j]
        
        self.h = [list(cluster[i]) for i in range(len(cluster))]
        self.determine_h_states(h_states)
        return cluster

    def determine_h_states(self, h_states):
        self.h_states = []
        for _ in range(len(self.h)):
            self.h_states = self.h_states + [h_states]
