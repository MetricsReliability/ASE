# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:02:48 2019

@author: ALIREZA HEDIEHLOO
mail: alirezahediehloo@gmail.com
"""

import numpy as np
import ctypes
from auto_structure import AutoStructure


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


# h_state determines the range of each latent variable in the network
class AmlmnbBase:
    def __init__(self, h=[], h_states=[], delta=1e-10, alpha=0, mode_h='individual'):
        self.pai_c_h = []
        self.pc = []
        self.ph = []
        self.I = []
        self.convergence = []
        self.compact_attribute = []
        self.h = h
        self.h_states = h_states
        self.L_all_h = 1
        for i in range(len(self.h_states)):
            self.L_all_h *= self.h_states[i]
        self.index_of_hs_atr = []
        self.delta = delta
        self.alpha = alpha
        self.mode_h = mode_h  # mode_h = 'individual' or 'joint'
        self.x_dim = 0
        self.y_dim = 0
        self.X = 0
        self.y = 0

    def auto_struct(self, x, y, m, sp, pairwise=False):
        auto_s = AutoStructure(x, y, m, pairwise, sp)
        auto_s.auto_structure(x, y)

    def fit(self, x_slow, y_slow, iteration=100):

        h_atr = []
        for i in range(len(x_slow[0])):
            h_atr = h_atr + [[]]
            for j in range(len(self.h)):
                if i in self.h[j]:
                    h_atr[i] = h_atr[i] + [j]
        hs_states = []
        for i in range(len(x_slow[0])):
            hs_states = hs_states + [1]
            for j in range(len(h_atr[i])):
                hs_states[i] *= self.h_states[h_atr[i][j]]

        self.X, self.y, self.compact_attribute = code(x_slow, y_slow)

        self.x_dim = len(y_slow)
        self.y_dim = len(x_slow[0])
        x_dim = ctypes.c_int(len(y_slow))
        y_dim = ctypes.c_int(len(x_slow[0]))

        # added by nima

        len_h = ctypes.c_int(len(self.h))
        len_h_i = (ctypes.c_int * len(self.h))(*[len(self.h[i]) for i in range(len(self.h))])
        h = (ctypes.POINTER(ctypes.c_int) * len(self.h))()
        for i in range(len(self.h)):
            h[i] = (ctypes.c_int * len(self.h[i]))(*self.h[i])

        h_states = (ctypes.c_int * len(self.h_states))(*self.h_states)

        iteration = ctypes.c_int(iteration)

        delta = ctypes.c_double(self.delta)

        alpha = ctypes.c_double(self.alpha)

        if self.mode_h == 'individual':
            mode_h = ctypes.c_int(1)
        else:
            mode_h = ctypes.c_int(0)

        len_compact_attribute = ctypes.c_int(len(self.compact_attribute))
        len_compact_attribute_i = (ctypes.c_int * len(self.compact_attribute))(*[
            len(self.compact_attribute[i]) for i in range(len(self.compact_attribute))])

        pc = (ctypes.c_double * len_compact_attribute_i[-1])()

        ph = (ctypes.c_double * self.L_all_h)()

        # pai_c_h[C=c][i][H=h][k] = P(Ai = ak | C=c,H=h)
        pai_c_h = (ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))) * len(pc))()
        for i in range(len(pc)):
            pai_c_h[i] = (ctypes.POINTER(ctypes.POINTER(ctypes.c_double)) * y_dim.value)()
            for j in range(y_dim.value):
                pai_c_h[i][j] = (ctypes.POINTER(ctypes.c_double) * hs_states[j])()
                for k in range(hs_states[j]):
                    pai_c_h[i][j][k] = (ctypes.c_double * len(self.compact_attribute[j]))()

        # index_of_hs_atr[feature][L_all_h]
        index_of_hs_atr = (ctypes.POINTER(ctypes.c_int) * y_dim.value)()
        for i in range(y_dim.value):
            index_of_hs_atr[i] = (ctypes.c_int * self.L_all_h)()

        convergence = (ctypes.c_double * iteration.value)()

        fit = ctypes.CDLL('E:\\apply\\york\\project\\source\\DLL_with_dev.dll')

        fit.fit(self.X, self.y, x_dim, y_dim, h, len_h, len_h_i, h_states,
                iteration, delta, alpha, mode_h,
                len_compact_attribute, len_compact_attribute_i,
                pc, ph, pai_c_h, index_of_hs_atr, convergence)

        self.pai_c_h = []
        for i in range(len(pc)):
            self.pai_c_h = self.pai_c_h + [[]]
            for j in range(y_dim.value):
                self.pai_c_h[i] = self.pai_c_h[i] + [[]]
                for k in range(hs_states[j]):
                    self.pai_c_h[i][j] = self.pai_c_h[i][j] + [[]]
                    for l in range(len(self.compact_attribute[j])):
                        self.pai_c_h[i][j][k] = self.pai_c_h[i][j][k] + [pai_c_h[i][j][k][l]]

        self.pc = []
        for i in range(len(pc)):
            self.pc = self.pc + [pc[i]]

        self.ph = []
        for i in range(self.L_all_h):
            self.ph = self.ph + [ph[i]]

        self.index_of_hs_atr = []
        for i in range(y_dim.value):
            self.index_of_hs_atr = self.index_of_hs_atr + [[]]
            for j in range(self.L_all_h):
                self.index_of_hs_atr[i] = self.index_of_hs_atr[i] + [index_of_hs_atr[i][j]]

        self.convergence = []
        for i in range(iteration.value):
            self.convergence = self.convergence + [convergence[i]]

    def continue_(self, iteration=100):
        h_atr = []
        for i in range(self.y_dim):
            h_atr = h_atr + [[]]
            for j in range(len(self.h)):
                if i in self.h[j]:
                    h_atr[i] = h_atr[i] + [j]
        hs_states = []
        for i in range(self.y_dim):
            hs_states = hs_states + [1]
            for j in range(len(h_atr[i])):
                hs_states[i] *= self.len_h[h_atr[i][j]]

        x_dim = ctypes.c_int(self.x_dim)
        y_dim = ctypes.c_int(self.y_dim)

        len_h = ctypes.c_int(len(self.h))
        len_h_i = (ctypes.c_int * len(self.h))(*[len(self.h[i]) for i in range(len(self.h))])
        h = (ctypes.POINTER(ctypes.c_int) * len(self.h))()
        for i in range(len(self.h)):
            h[i] = (ctypes.c_int * len(self.h[i]))(*self.h[i])

        h_states = (ctypes.c_int * len(self.h_states))(*self.h_states)

        iteration = ctypes.c_int(iteration)

        delta = ctypes.c_double(self.delta)

        alpha = ctypes.c_double(self.alpha)

        if self.mode_h == 'individual':
            mode_h = ctypes.c_int(1)
        else:
            mode_h = ctypes.c_int(0)

        len_compact_attribute = ctypes.c_int(len(self.compact_attribute))
        len_compact_attribute_i = (ctypes.c_int * len(self.compact_attribute))(*[
            len(self.compact_attribute[i]) for i in range(len(self.compact_attribute))])

        pc = (ctypes.c_double * len_compact_attribute_i[-1])()

        ph = (ctypes.c_double * self.L_all_h)()
        for i in range(self.L_all_h):
            ph[i] = self.ph[i]

        # pai_c_h[C=c][i][H=h][k] = P(Ai = ak | C=c,H=h)
        pai_c_h = (ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))) * len(pc))()
        for i in range(len(pc)):
            pai_c_h[i] = (ctypes.POINTER(ctypes.POINTER(ctypes.c_double)) * y_dim.value)()
            for j in range(y_dim.value):
                pai_c_h[i][j] = (ctypes.POINTER(ctypes.c_double) * hs_states[j])()
                for k in range(hs_states[j]):
                    pai_c_h[i][j][k] = (ctypes.c_double * len(self.compact_attribute[j]))()
                    for l in range(len(self.compact_attribute[j])):
                        pai_c_h[i][j][k][l] = self.pai_c_h[i][j][k][l]

        # index_of_hs_atr[feature][L_all_h]
        index_of_hs_atr = (ctypes.POINTER(ctypes.c_int) * y_dim.value)()
        for i in range(y_dim.value):
            index_of_hs_atr[i] = (ctypes.c_int * self.L_all_h)()

        convergence = (ctypes.c_double * iteration.value)()

        fit = ctypes.CDLL('E:\\apply\\york\\project\\source\\DLL_with_dev.dll')

        fit.continue_(self.X, self.y, x_dim, y_dim, h, len_h, len_h_i, h_states,
                      iteration, delta, alpha, mode_h,
                      len_compact_attribute, len_compact_attribute_i,
                      pc, ph, pai_c_h, index_of_hs_atr, convergence)

        self.pai_c_h = []
        for i in range(len(pc)):
            self.pai_c_h = self.pai_c_h + [[]]
            for j in range(y_dim.value):
                self.pai_c_h[i] = self.pai_c_h[i] + [[]]
                for k in range(hs_states[j]):
                    self.pai_c_h[i][j] = self.pai_c_h[i][j] + [[]]
                    for l in range(len(self.compact_attribute[j])):
                        self.pai_c_h[i][j][k] = self.pai_c_h[i][j][k] + [pai_c_h[i][j][k][l]]

        self.pc = []
        for i in range(len(pc)):
            self.pc = self.pc + [pc[i]]

        self.ph = []
        for i in range(self.L_all_h):
            self.ph = self.ph + [ph[i]]

        self.index_of_hs_atr = []
        for i in range(y_dim.value):
            self.index_of_hs_atr = self.index_of_hs_atr + [[]]
            for j in range(self.L_all_h):
                self.index_of_hs_atr[i] = self.index_of_hs_atr[i] + [index_of_hs_atr[i][j]]

        for i in range(iteration.value):
            self.convergence = self.convergence + [convergence[i]]

    def predict(self, x):
        pr_y = []
        for i in range(len(x)):
            # c = P(sample member of C)
            p_s_c = []
            max_ = []
            for j in range(len(self.pc)):
                max_ = max_ + [0]
                p_s_c = p_s_c + [[]]
                for k in range(self.L_all_h):
                    p_s_c[j] = p_s_c[j] + [self.ph[k]]
                    for l in range(len(self.pai_c_h[0])):
                        if x[i][l] in self.compact_attribute[l]:
                            index = self.compact_attribute[l].index(x[i][l])
                            # pai_c_h[C=c][i][H=h][k] = P(Ai = ak | C=c,H=h)
                            p_s_c[j][k] += self.pai_c_h[j][l][self.index_of_hs_atr[l][k]][index]
                        else:
                            p_s_c[j][k] += np.log2(1 / len(self.pc))
                max_[j] = max(p_s_c[j])
            temp = max(max_)
            sums = []
            for j in range(len(self.pc)):
                sums = sums + [0]
                for k in range(self.L_all_h):
                    p_s_c[j][k] = 2 ** (p_s_c[j][k] - temp)
                sums[j] = sum(p_s_c[j])
                sums[j] = np.log2(sums[j])
            pr_y = pr_y + [sums]
        c = self.compact_attribute[-1]
        pred = []
        for i in range(len(pr_y)):
            pred.append(c[pr_y[i].index(max(pr_y[i]))])
        # return pr_y, p_s_c, c, pred
        return pred
