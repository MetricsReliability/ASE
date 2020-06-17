import csv
import math

import pandas as pd
from collections import OrderedDict
import sys
import numpy as np


class MLMNB:
    def __init__(self, setting):
        self.attrib = setting['num_attribute']
        self.sample = setting['num_sample']
        self.s_class = setting['class_size']
        self.class_index = setting['class_index']
        self.latent_size = setting['latent_size']
        self.attrib_size = setting['attribute_size']

    def count_duplicate_row_occurrence(self, data, union_data):
        mylist = []
        for i in range(0, len(data)):
            mylist.append(str(data[i, :]))
        dictionary = OrderedDict()
        for item in mylist:
            if item in dictionary:
                dictionary[item] += 1
            else:
                dictionary[item] = 1
        index_holder = np.zeros([len(union_data), 1])
        i = 0
        for key, val in dictionary.items():
            index_holder.put(i, val)
            i = i + 1
        return index_holder

    def compress_trainingData(self, data):
        newdata = pd.DataFrame(data)
        union_data = pd.concat([newdata, newdata], ignore_index=True).drop_duplicates().reset_index(drop=True)

        count_list = self.count_duplicate_row_occurrence(data, union_data)
        count_list = np.array(count_list)

        union_data = pd.DataFrame.as_matrix(union_data)
        compressed_data = np.concatenate((count_list, union_data), axis=1)
        return compressed_data

    def initializeProbabilities(self, data):

        theta = [[], [], []]

        hist_c = np.histogram(data[:, -1], self.s_class)
        a = len(hist_c[0])
        pclass = hist_c[0] / len(data)
        for i in range(len(pclass)):
            if pclass[i] == 0:
                pclass[i] = sys.float_info.epsilon

        theta[0].append(pclass)

        ph = [np.random.uniform(0.45, 0.55) for _ in range(self.latent_size)]
        ph = np.divide(ph, np.sum(ph))
        theta[1].append(np.array(ph))

        for i in range(self.attrib):
            probs = np.random.uniform(0.45, 0.55,
                                      size=(self.s_class, self.latent_size, int(self.attrib_size[i])))
            s = np.sum(probs, 2)
            self.ndim = probs.shape
            for j in range(np.size(probs, 2)):
                probs[:, :, j] = probs[:, :, j] / s
            theta[2].append(probs)

        return theta

    def expectation(self, init_params, compressed_data):

        sd = len(compressed_data)
        CPH = np.zeros((sd, self.latent_size))
        cd = compressed_data
        ll = 0

        for i in range(sd):
            for j in range(self.latent_size):
                temp = 1
                for k in range(self.attrib):
                    pattrib = init_params[2][k]
                    temp = temp * pattrib[int(cd[i, -1]) - 1, int(j), int(cd[i, k + 1]) - 1]
                pc = init_params[0]
                ph = init_params[1]

                temp = temp * pc[0][int(cd[i, -1]) - 1] * ph[0][j]
                CPH[i][j] = temp
            if np.sum(CPH[i][:]) == 0:
                CPH[i][:] = CPH[i][:] + 0.000001
            ll = np.log(sum(CPH[i][:]))
            CPH[i][:] = CPH[i][:] / np.sum(CPH[i][:])
        # for jj in range(len(CPH)):
        #     ll = ll + np.log(sum(CPH[jj][:]))
        return CPH, ll

    def maximization(self, cd, init_params):
        theta = init_params

        ph = init_params[1][0]
        for j in range(self.latent_size):
            ph[j] = np.sum(np.multiply(cd[:, 0], cd[:, 1 + self.attrib + j + 1]))

        ph = ph / np.sum(ph)
        theta[1][0] = ph

        for k in range(len(theta[2])):
            pxk = init_params[2][k]
            for c in range(self.s_class):
                for h in range(self.latent_size):
                    for i in range(int(self.attrib_size[k])):
                        pxk[c, h, i] = np.sum(
                            np.multiply(cd[np.where((cd[:, k + 1] == i + 1) & (
                                    cd[:, 1 + self.attrib] == c + 1)), 0],
                                        cd[np.where(
                                            (cd[:, k + 1] == i + 1) & (cd[:,
                                                                       1 + self.attrib] == c + 1)), 2 + self.attrib + h]))
                    pxk[c, h, :] = pxk[c, h, :] + 0.09
                    pxk[c, h, :] = pxk[c, h, :] / np.sum(pxk[c, h, :])
            theta[2][k] = pxk
        return init_params

    def expectationMaximization(self, raw_data):
        compressed_data = self.compress_trainingData(raw_data)
        _params = self.initializeProbabilities(raw_data)

        prevll = math.inf
        converged = 0
        iter = 0
        # while not converged:
        for i in range(10):
            iter = iter + 1
            CPH, ll = self.expectation(_params, compressed_data)
            complete_data = np.concatenate((compressed_data, CPH), axis=1)
            _params = self.maximization(complete_data, _params)
            print('EM ITERATION:', iter, 'LOG LIKELIHOOD:', ll)

            error = np.abs(ll - prevll)
            if error <= sys.float_info.epsilon:
                converged = 1
            prevll = ll

        return _params

    def inference(self, params, testdata):
        pc = params[0]
        ph = params[1]
        class_prob = np.zeros((self.s_class, 1), dtype=np.longfloat)
        ak = []
        for c in range(self.s_class):
            temp = 1
            for h in range(self.latent_size):
                for k in range(self.attrib):
                    pxj = params[2][k]
                    # temp = temp * pxj[c, h, int(testdata[k]) - 1]
                    temp = temp * pxj[c, h, int(testdata[k]) - 1]
                # temp = temp * pc[0][c] * ph[0][h]
                temp = temp * pc[0][c] * ph[0][h]
        #     ak.append(np.log2(temp))
        # A = max(ak)
        # res = A + np.log2(sum(np.exp(ak - A)))
        # class_prob[0, 0] = 2 ** res
        # class_prob[1, 0] = 1 - 2 ** res
        # temp = 2 ** temp
            class_prob[c, 0] = temp + class_prob[c, 0]
        return class_prob

    def writeCSV(self, dataObj, filename):
        with open(filename, 'w', newline='') as csvfile:
            wr = csv.writer(csvfile)
            for val in dataObj:
                wr.writerow(val)
                # wr.write(val)
            csvfile.close()


def prepare_new_dataset(old_data, indices):
    array = np.zeros((np.size(old_data, 0), np.size(indices, 1) + 1))
    for i in range(np.size(indices, 1)):
        array[:, i] = old_data[:, indices[0][i]]
    array[:, -1] = old_data[:, -1]
    return array


def get_attribSize(data):
    size_holder = []
    for i in range(np.size(data, 1)):
        size_holder.append(len(np.unique(data[:, i])))
    return size_holder

#
# def main():
#     setting = {}
#     key_run = 10
#     key_fold = 10
#     perf_obj = PerformanceEvaluation()
#
#     input_path = "C:\\Users\\Nima\\Desktop\\letter"
#     output_path = "C:\\Users\\Nima\\Desktop\\letter\\mytest.csv"
#
#     os.chdir(input_path)
#     output_format_extension = 'csv'
#
#     _dataList = [i for i in glob.glob('*.{}'.format(output_format_extension))]
#     result_holder = []
#     for i in range(len(_dataList)):
#         data = np.genfromtxt(_dataList[i], delimiter=",")
#         # cmi_obj = CMI(data)
#         # selected_index = cmi_obj.calc_cmi(data)
#
#         # data = prepare_new_dataset(data, selected_index)
#         node_size = get_attribSize(data)
#         m, N = np.shape(data)
#
#         setting['class_size'] = node_size[-1]
#         setting['num_attribute'] = N - 1
#         setting['num_sample'] = m
#         setting['class_index'] = N
#         setting['attribute_size'] = node_size[0:N-1]
#         subset_size = round(len(data) / key_fold)
#         for ii in range(2, 3):
#             setting['latent_size'] = ii
#             mlnb_obj = MLMNB(setting)
#             for j in range(key_run):
#                 print(j)
#                 index_holder = 0
#                 for k in range(key_fold):
#                     index = subset_size * (k + 1)
#                     test_set = data[index_holder:index]
#                     data = np.delete(data, np.s_[index_holder: index], axis=0)
#
#                     params, _ = mlnb_obj.expectationMaximization(data)
#
#                     prob = np.zeros((setting['class_size'], 1))
#
#                     for counter in range(len(test_set)):
#                         a = mlnb_obj.inference(params, test_set[counter, :])
#                         prob = np.concatenate((prob, a), axis=1)
#                     prob = np.delete(prob, 0, 1)
#
#                     pred = []
#                     for v in range(np.size(prob, 1)):
#                         pred.append(np.argmax(prob[:, v]) + 1)
#
#                     y_true = test_set[:, N - 1]
#                     y_pred = pred
#
#                     perf_holder = perf_obj.compute_measures(y_true, y_pred)
#
#                     data = np.concatenate((data, test_set), axis=0)
#                     index_holder = index
#
#                     serialized_data = perf_obj.Serializer(
#                         str(_dataList[i]), j, k, 'MLMNB' + str(ii), perf_holder)
#
#                     result_holder.append(serialized_data)
#         mlnb_obj.writeCSV(result_holder, output_path)
#
#
# if __name__ == '__main__':
#     main()
