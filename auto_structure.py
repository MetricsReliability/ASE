import numpy as np
import ctypes
import itertools
from scipy import stats
import math
import heapq


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


def selection_sort(pval, index):
    n = len(pval)
    for i in range(n):
        jmin = i
        for j in range(i + 1, n):
            if pval[j] < pval[jmin]:
                jmin = j
        if jmin != i:
            temp1 = pval[i]
            temp2 = index[i]
            pval[i] = pval[jmin]
            index[i] = index[jmin]
            pval[jmin] = temp1
            index[jmin] = temp2
    return pval, index


def is2vct_equal(a, b):
    for i in range(len(a)):
        if a[i] != b[i]:
            return 0
    return 1


class AutoStructure():
    def __init__(self, x_slow, y_slow, m, pairwise, sp):
        self.I = []
        self.h = []
        self.sp = sp
        self.pairwise_mode = pairwise
        self.num_records = m
        self.h_states = []
        x, y, self.compact_attribute = code(x_slow, y_slow)
        x_dim = len(y_slow)
        y_dim = len(x_slow[0])
        self.mutual = (ctypes.POINTER(ctypes.c_double) * y_dim)()
        for i in range(y_dim):
            self.mutual[i] = (ctypes.c_double * y_dim)()
        x_dim = ctypes.c_int(x_dim)
        y_dim = ctypes.c_int(y_dim)

        imc = ctypes.CDLL('./find_mutual_conditional.dll')
        imc.information_mutual_conditional_all(x, y, x_dim, y_dim, self.mutual)

        y_dim = y_dim.value
        for i in range(y_dim):
            self.I = self.I + [[]]
            for j in range(y_dim):
                self.mutual[i][j] = self.mutual[i][j] * 2 * self.num_records
                self.I[i] = self.I[i] + [self.mutual[i][j]]

    def auto_structure(self, x_slow, y_slow):
        self.p = np.zeros((len(x_slow[0]), len(x_slow[0])))
        N = math.trunc(len(x_slow[0]) / 2)
        p_value_vec = []
        index_vec = []
        for i in range(len(x_slow[0])):
            for j in range(len(x_slow[0])):
                if i != j:
                    d = (len(self.compact_attribute[i]) - 1) * (len(self.compact_attribute[j]) - 1) * 2
                    p_val = stats.chi2.pdf(self.mutual[i][j], d)
                    self.p[i][j] = p_val

        latent_variables = []
        state_space = []
        for i in range(len(x_slow[0])):
            for j in range(len(x_slow[0])):
                if i < j:
                    p_value_vec.append(self.p[i][j])
                    index_vec.append([i, j])
        [p_val, index] = selection_sort(p_value_vec, index_vec)

        k = 0
        for i, _ in reversed(list(enumerate(p_val))):
            latent_variables.append(index[i])
            state_space.append(self.sp)
            k += 1
            if k >= N:
                break
        latent_tuples = [tuple(item) for item in latent_variables]
        p1 = []
        p2 = []
        for v1, v2 in latent_tuples:
            p1.append(v1)
            p2.append(v2)
        p1 = set(p1)
        p2 = set(p2)
        new_latent_list = []
        new_state_space = []
        if len(p1) != len(p2):
            for i in range(min(len(p1), len(p2))):
                new_latent_list.append([list(p1)[i], list(p2)[i]])
            self.h = new_latent_list
            for i in range(len(new_latent_list)):
                new_state_space.append(self.sp)
            self.h_states = new_state_space
        else:
            self.h = latent_variables
            self.h_states = state_space
        return self
