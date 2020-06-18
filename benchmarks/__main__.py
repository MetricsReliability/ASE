import random
import time
from multiprocessing import Process, Manager

from sklearn.utils import shuffle
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, \
    roc_auc_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from data_collection_manipulation.data_handler import IO
from configuration_files.setup_config import LoadConfig
from data_collection_manipulation.data_handler import DataPreprocessing
from sklearn import tree
import pandas as pd
from libs.MLMNB_package_python import MLMNB
from sklearn.preprocessing import KBinsDiscretizer
import libs.entropy_estimators as ee
from scipy import stats

gnb_obj = GaussianNB()
rnd_obj = RandomForestClassifier()
dh_obj = IO()

disc = KBinsDiscretizer(n_bins=10, encode='ordinal')
from latent_base import AmlmnbBase
from latents import AMLMNB


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


def feature_selection(tr, ts):
    [m, N] = np.shape(tr)
    tr_cmi = np.zeros((N - 1, N - 1))

    single_var_cmi = np.zeros((1, N - 1))
    p_store_tr_single = np.zeros((1, N - 1))

    p_store_tr = np.zeros((N - 1, N - 1))
    p_store_ts = np.zeros((N - 1, N - 1))

    u_tr = []
    u_ts = []
    for k in range(0, N - 1):
        unique1, counts1 = np.unique(tr[:, k], return_counts=True)
        u_tr.append(len(unique1))

    for i in range(N - 1):
        for j in range(N - 1):
            if i != j:
                tr_cmi[i, j] = ee.cmidd(tr[:, i], tr[:, j], tr[:, -1])
    tr_cmi = 2 * m * tr_cmi
    ############################################################
    for n in range(N - 1):
        single_var_cmi[0, n] = ee.midd(tr[:, n], tr[:, -1])
    single_var_cmi = 2 * m * single_var_cmi

    for n in range(N - 1):
        d1 = 2 * (u_tr[n] - 1)
        p_store_tr_single[0][n] = stats.chi2.pdf(single_var_cmi[0, n], d1)
    single_var_cmi = np.argsort(single_var_cmi)
    selected_idx = []

    idx = np.size(single_var_cmi, 1)
    k = 0
    while k != 5:
        selected_idx.append(single_var_cmi[0, idx - 1])
        idx -= 1
        k += 1
    ################################################################
    for j in range(N - 1):
        for i in range(N - 1):
            if i != j:
                d1 = 2 * (u_tr[j] - 1) * (u_tr[i] - 1)
                d2 = (u_ts[j] - 1) + (u_ts[i] - 1)
                p_store_tr[j][i] = stats.chi2.cdf(tr_cmi[j, i], d1)

    latent_variables = []
    state_space = []
    p_value_vec_tr = []
    index_vec_tr = []
    p_value_vec_ts = []
    index_vec_ts = []
    for i in range(N - 1):
        for j in range(N - 1):
            if i < j:
                p_value_vec_tr.append(p_store_tr[i][j])
                index_vec_tr.append([i, j])
                index_vec_ts.append([i, j])
    [p_val_tr, index_tr] = selection_sort(p_value_vec_tr, index_vec_tr)

    k = 0
    for i in range(len(p_val_tr)):
        latent_variables.append(index_tr[i])
        k += 1
        if k >= 10:
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
    p1 = list(p1)
    p1.append(N - 1)
    # p1 is CMI(X,Y|C)
    # selected_idx is CMI(X_i|C)
    selected_idx.append(N - 1)
    tr = tr[:, selected_idx]
    return tr


class PerformanceEvaluation:
    def __init__(self, configuration):
        self.config = configuration
        self.recall_flag = False
        self.precision_flag = False
        self.F1_flag = False
        self.ACC_flag = False
        self.MCC_flag = False
        self.AUC_flag = False

        for measure in self.config['evaluation_measures']:
            if measure == 'Precision':
                self.precision_flag = True
            if measure == 'Recall':
                self.recall_flag = True
            if measure == 'F1':
                self.F1_flag = True
            if measure == 'ACC':
                self.ACC_flag = True
            if measure == 'MCC':
                self.MCC_flag = True
            if measure == 'AUC':
                self.AUC_flag = True

    def compute_measures(self, actual, pred):
        perf_pack = []

        a = classification_report(actual, pred, output_dict=True)
        if self.precision_flag:
            perf_pack.append(round(a['weighted avg']['precision'], 2))
        if self.recall_flag:
            perf_pack.append(round(a['weighted avg']['recall'], 2))
        if self.F1_flag:
            perf_pack.append(round(a['weighted avg']['f1-score'], 2))
        if self.ACC_flag:
            ACC = accuracy_score(actual, pred)
            perf_pack.append(round(ACC, 2))
        if self.MCC_flag:
            MCC = matthews_corrcoef(actual, pred, sample_weight=None)
            perf_pack.append(round(MCC, 2))
        if self.AUC_flag:
            _auc = roc_auc_score(actual, pred, average=None)
            perf_pack.append(round(roc_auc_score(actual, pred, average='weighted'), 2))
        return perf_pack


class Benchmarks:
    def __init__(self, dataset, dataset_names, file_names, _configuration):
        self.dataset = dataset
        self.config = _configuration
        self.dataset_names = dataset_names
        self.dataset_file_names = file_names
        self.model_holder = self.config['defect_models']

        if self.config['cross_validation_type'] == 1:
            self.validator = LeaveOneOut()
        elif self.config['cross_validation_type'] == 2:
            self.validator = StratifiedKFold(n_splits=10, random_state=2, shuffle=True)
        elif self.config['cross_validation_type'] == 3:
            self.validator = KFold(n_splits=self.config['number_of_folds'])
        # n_estimators=1, bootstrap=False, n_jobs=-1, max_depth=3, max_features=None,
        #                                    random_state=99
        self.classifiers = [
            RandomForestClassifier(),
            GaussianNB(), LogisticRegression(), tree.DecisionTreeClassifier(), svm.SVC(),
            KNeighborsClassifier(n_neighbors=3)]

        self.header2 = ["Defect models", "projects' category", "Tr version", "Ts version", "Iterations",
                        "Precision", "Recall", "F1 Score", "ACC", "MCC", "AUC"]

        self.header1 = ["File name", "original status", "predicted status"]

        self.cv_header = ["Key_Dataset", "Key_Run", "Key_Fold", "Key_Scheme", "Precsion", "Recall", "F1_Score", "ACC",
                          "MCC", "AUC"]

        self.perf_obj = PerformanceEvaluation(self.config)

        self.temp_addr = "E:\\apply\\york\\project\\source\\outputs\\file_level" \
                         "\\different_releases_tr_ts\\res_jmt.csv"

    def differen_release_MLMNB(self):
        binning = True
        self.config_MLMNB = {}
        self.dataset = DataPreprocessing.binerize_class(self.dataset)
        temp_result = [self.header2]
        for iterations in range(self.config['iterations']):
            for latent_size in range(1, 8):
                print(latent_size)
                self.config_MLMNB['latent_size'] = latent_size
                for ds_cat, ds_val in self.dataset.items():
                    for i in range(len(ds_val)):
                        for j in range(i + 1, len(ds_val)):
                            print(ds_cat)
                            tr = np.array(ds_val[j])
                            ts = np.array(ds_val[i])

                            # tr = feature_selection(tr)

                            learner = AmlmnbBase(h=[], h_states=[latent_size], delta=0.01, alpha=0.000001)

                            if binning:
                                # tr[:, 0:-1] = disc.fit_transform(tr[:, 0:-1])
                                # ts[:, 0:-1] = disc.fit_transform(ts[:, 0:-1])
                                # 3, 4, 5, 6, 9, 10, 11, 13, 14, 17, 19]
                                reduced_tr = np.delete(tr, [9, 11, 13, 14, 17, 19], axis=1)
                                reduced_ts = np.delete(ts, [9, 11, 13, 14, 17, 19], axis=1)
                                discretized_tr = disc.fit_transform(tr[:, [9, 11, 13, 14, 17, 19]])
                                discretized_ts = disc.fit_transform(ts[:, [9, 11, 13, 14, 17, 19]])
                                tr = np.concatenate((discretized_tr, reduced_tr), axis=1)
                                ts = np.concatenate((discretized_ts, reduced_ts), axis=1)
                            whole_data = np.concatenate((tr, ts), axis=0)
                            learner.fit(tr[:, 0:-1], tr[:, -1], 100)
                            nominal_pred = learner.predict(ts[:, 0:-1])
                            # self.s_attrib = DataPreprocessing.get_metrics_size(data=whole_data)
                            # [m, N] = np.shape(tr)
                            # self.config_MLMNB['class_size'] = self.s_attrib[-1]
                            # self.config_MLMNB['num_attribute'] = N - 1
                            # self.config_MLMNB['num_sample'] = m
                            # self.config_MLMNB['class_index'] = N
                            # self.config_MLMNB['attribute_size'] = self.s_attrib[0:-1]
                            # self.mlmnb_obj = MLMNB(self.config_MLMNB)
                            # params = self.mlmnb_obj.expectationMaximization(tr)

                            # prob = np.zeros((self.config_MLMNB['class_size'], 1), dtype=np.longfloat)

                            # for counter in range(len(ts)):
                            #     a = self.mlmnb_obj.inference(params, ts[counter, :])
                            #     prob = np.concatenate((prob, a), axis=1)
                            # prob = np.delete(prob, 0, 1)

                            # y_pred = []
                            # for v in range(np.size(prob, 1)):
                            #     y_pred.append(np.argmax(prob[:, v]) + 1)
                            # y_pred = np.array(y_pred)
                            # y_true = ts[:, N - 1]

                            y_true = ts[:, -1]
                            y_pred = np.array(nominal_pred)

                            perf_holder = self.perf_obj.compute_measures(y_true, y_pred)

                            release_pack = ["MLMNB" + str(latent_size), ds_cat, self.dataset_names[ds_cat][i],
                                            self.dataset_names[ds_cat][j], iterations,
                                            *perf_holder]

                            a = pd.concat(
                                [self.dataset_file_names[ds_cat][j], pd.DataFrame(y_true.reshape((-1, 1))).reindex(
                                    self.dataset_file_names[ds_cat][j].index),
                                 pd.DataFrame(y_pred.reshape((-1, 1))).reindex(
                                     self.dataset_file_names[ds_cat][j].index)], axis=1, ignore_index=True)

                            a.columns = self.header1
                            temp_result.append(release_pack)

                            addr = self.temp_addr + "MLMNB" + str(latent_size) + "_" + "Iteration {}".format(
                                iterations) + "_" + self.dataset_names[ds_cat][j]

                            # pd.DataFrame.to_csv(a, path_or_buf=addr, sep=',')
                            dh_obj.write_csv(temp_result, self.config['file_level_different_release_results_whole'])
        return None

    def different_release(self):
        binning = True
        self.config_MLMNB = {}
        self.dataset = DataPreprocessing.binerize_class(self.dataset)
        temp_result = [self.header2]
        for model_name, clf in zip(self.model_holder, self.classifiers):
            for ds_cat, ds_val in self.dataset.items():
                for i in range(len(ds_val)):
                    for j in range(i + 1, len(ds_val)):
                        ## think about attrib size for different releases.
                        # ds_val[i][:] = np.nan_to_num(ds_val[i])
                        # ds_val[j][:] = np.nan_to_num(ds_val[j])
                        tr = np.array(ds_val[i])
                        ts = np.array(ds_val[j])

                        if binning:
                            tr[:, 0:-1] = disc.fit_transform(tr[:, 0:-1])
                            ts[:, 0:-1] = disc.fit_transform(ts[:, 0:-1])
                            # reduced_tr = np.delete(tr, [3, 4, 5, 6, 9, 10, 11, 13, 14, 17, 19], axis=1)
                            # reduced_ts = np.delete(ts, [3, 4, 5, 6, 9, 10, 11, 13, 14, 17, 19], axis=1)
                            # discretized_tr = disc.fit_transform(tr[:, [3, 4, 5, 6, 9, 10, 11, 13, 14, 17, 19]])
                            # discretized_ts = disc.fit_transform(ts[:, [3, 4, 5, 6, 9, 10, 11, 13, 14, 17, 19]])
                            # tr = np.concatenate((discretized_tr, reduced_tr), axis=1)
                            # ts = np.concatenate((discretized_ts, reduced_ts), axis=1)

                        self.s_attrib = DataPreprocessing.get_metrics_size(data=tr)

                        X_train = tr[:, 0:-1]
                        y_train = tr[:, -1]
                        X_test = ts[:, 0:-1]
                        y_test = ts[:, -1]
                        for iterations in range(self.config['iterations']):
                            clf.fit(X_train, y_train)
                            random.seed(100)
                            y_pred = clf.predict(X_test)

                            print(self.dataset_names[ds_cat][i])
                            perf_holder = self.perf_obj.compute_measures(y_test, y_pred)

                            release_pack = [model_name, ds_cat, self.dataset_names[ds_cat][i],
                                            self.dataset_names[ds_cat][j], iterations,
                                            *perf_holder]

                            a = pd.concat(
                                [self.dataset_file_names[ds_cat][j], pd.DataFrame(y_test.reshape((-1, 1))).reindex(
                                    self.dataset_file_names[ds_cat][j].index),
                                 pd.DataFrame(y_pred.reshape((-1, 1))).reindex(
                                     self.dataset_file_names[ds_cat][j].index)], axis=1, ignore_index=True)

                            a.columns = self.header1
                            temp_result.append(release_pack)

                            addr = self.temp_addr + model_name + "_" + "Iteration {}".format(
                                iterations) + "_" + self.dataset_names[ds_cat][j]

                            # pd.DataFrame.to_csv(a, path_or_buf=addr, sep=',')

            dh_obj.write_csv(temp_result, self.config['file_level_different_release_results_whole'])

    def delete_unused_NASA_metrics(self, _dataset, ds_cat):
        if ds_cat == "CM1":
            reduced_dataset = np.delete(_dataset, [7, 9, 11, 14, 17, 18, 19, 20, 22, 23, 24, 25, 29, 35], axis=1)
            discretized_tr = disc.fit_transform(_dataset[:, [7, 9, 11, 14, 17, 18, 19, 20, 22, 23, 24, 25, 29, 35]])
            _dataset = np.concatenate((discretized_tr, reduced_dataset), axis=1)
        if ds_cat == "JM1":
            reduced_dataset = np.delete(_dataset, [8, 9, 10, 11, 13, 14, 15], axis=1)
            discretized_tr = disc.fit_transform(_dataset[:, [8, 9, 10, 11, 13, 14, 15]])
            _dataset = np.concatenate((discretized_tr, reduced_dataset), axis=1)
        if ds_cat == "KC1" or ds_cat == "KC2" or ds_cat == "KC3":
            reduced_dataset = np.delete(_dataset, [5, 6, 7, 8, 9, 10, 11], axis=1)
            discretized_tr = disc.fit_transform(_dataset[:, [5, 6, 7, 8, 9, 10, 11]])
            _dataset = np.concatenate((discretized_tr, reduced_dataset), axis=1)
        return _dataset

    def cross_validation(self):
        temp_result = [self.cv_header]
        self.model_holder.append("MLMNB")
        children = []
        learner = AmlmnbBase(h=[], h_states=[3], delta=0.00001, alpha=0.02)
        self.classifiers.append(learner)
        temp = self.classifiers[0]
        self.classifiers[0] = learner
        self.classifiers[-1] = temp

        temp = self.model_holder[0]
        self.model_holder[0] = "MLMNB"
        self.model_holder[-1] = temp

        self.dataset = DataPreprocessing.binerize_class(self.dataset)
        for model_name, clf in zip(self.model_holder, self.classifiers):
            for ds_cat, ds_val in self.dataset.items():
                for i in range(len(ds_val)):
                    _dataset = np.array(ds_val[i])
                    # delete
                    if model_name == "MLMNB":
                        _dataset = feature_selection(_dataset)
                        for k in range(np.size(_dataset, 1) - 1):
                            children.append(k)
                        learner = AmlmnbBase(h=[children], h_states=[5], delta=0.01, alpha=0.01)
                        self.classifiers[0] = learner
                        _dataset[:, 0:-1] = disc.fit_transform(_dataset[:, 0:-1])
                    else:
                        if ds_cat == "CM1" or ds_cat == "JM1" or ds_cat == "KC1" or ds_cat == "KC2" or ds_cat == "KC3":
                            _dataset = self.delete_unused_NASA_metrics(_dataset, ds_cat)
                        else:
                            reduced_tr = np.delete(_dataset, [9, 11, 13, 14, 17, 19], axis=1)
                            discretized_tr = disc.fit_transform(_dataset[:, [9, 11, 13, 14, 17, 19]])
                            _dataset = np.concatenate((discretized_tr, reduced_tr), axis=1)
                    for key_iter in range(self.config['iterations']):
                        X = _dataset[:, 0:-1]
                        y = _dataset[:, -1]
                        k = 0
                        for train_idx, test_idx in self.validator.split(X, y):
                            print('CLASSIFIER:', model_name, "DATASET", self.dataset_names[ds_cat][i], 'ITERATION:',
                                  key_iter, 'CV_FOLD:', k)
                            X_train, X_test = X[train_idx], X[test_idx]
                            # y_test = actual class label of test data
                            # y_train = actual class label of train data
                            y_train, y_test = y[train_idx], y[test_idx]

                            X_train, y_train = shuffle(X_train, y_train)
                            clf.fit(X_train, y_train)
                            score = clf.predict(X_test)
                            perf_holder = self.perf_obj.compute_measures(y_test, score)
                            cross_val_pack = [str(self.dataset_names[ds_cat][i]), key_iter, k, model_name,
                                              *perf_holder]

                            k = k + 1

                            temp_result.append(cross_val_pack)
                            dh_obj.write_csv(temp_result,
                                             self.config['file_level_WPDP_cross_validation_results_des'])


def cv_loop(self, clf, X, y, k):
    precision = []
    recall = []
    f1 = []
    acc = []
    mcc = []
    auc = []
    for train_idx, test_idx in self.validator.split(X, y):
        # learner.auto_structure(x_train, y_train)
        X_train, X_test = X[train_idx], X[test_idx]
        # y_test = actual class label of test data
        # y_train = actual class label of train data
        y_train, y_test = y[train_idx], y[test_idx]

        X_train, y_train = shuffle(X_train, y_train)

        clf.fit(X_train, y_train)
        score = clf.predict(X_test)
        perf_holder = self.perf_obj.compute_measures(y_test, score)
        precision += [perf_holder[0]]
        recall += [perf_holder[1]]
        f1 += [perf_holder[2]]
        acc += [perf_holder[3]]
        mcc += [perf_holder[4]]
        auc += [perf_holder[5]]
    return precision, recall, f1, acc, mcc, auc


def worker_thread(self, clf, X, y, k, jj, precision_res, recall_res, f1_res, accuracy_res, mcc_res, auc_res):
    precision, recall, f1, acc, mcc, auc = self.cv_loop(clf, X, y, k)
    for i in range(k):
        precision_res[jj * k + i] = precision[i]
        recall_res[jj * k + i] = recall[i]
        f1_res[jj * k + i] = f1[i]
        accuracy_res[jj * k + i] = acc[i]
        mcc_res[jj * k + i] = mcc[i]
        auc_res[jj * k + i] = auc[i]


def cv_parallel(self, cv_iteration):
    if __name__ == '__main__':
        k = self.config["number_of_folds"]
        manager = Manager()
        precision = manager.list([0] * cv_iteration * k)
        recall = manager.list([0] * cv_iteration * k)
        f1 = manager.list([0] * cv_iteration * k)
        accuracy = manager.list([0] * cv_iteration * k)
        mcc = manager.list([0] * cv_iteration * k)
        auc = manager.list([0] * cv_iteration * k)
        temp_result = [self.cv_header]
        self.model_holder.append("MLMNB")
        children = []
        self.dataset = DataPreprocessing.binerize_class(self.dataset)
        for model_name, clf in zip(self.model_holder, self.classifiers):
            for ds_cat, ds_val in self.dataset.items():
                for i in range(len(ds_val)):
                    for k in range(np.size(ds_val[i], 1) - 1):
                        children.append(k)
                    model = AmlmnbBase(h=[children], h_states=[10], delta=0.01, alpha=0.00001)
                    self.classifiers.append(model)
                    _dataset = np.array(ds_val[i])
                    if ds_cat == "CM1" or ds_cat == "JM1" or ds_cat == "KC1" or ds_cat == "KC2" or ds_cat == "KC3":
                        _dataset = self.delete_unused_NASA_metrics(_dataset, ds_cat)
                    else:
                        reduced_tr = np.delete(_dataset, [9, 11, 13, 14, 17, 19], axis=1)
                        discretized_tr = disc.fit_transform(_dataset[:, [9, 11, 13, 14, 17, 19]])
                        _dataset = np.concatenate((discretized_tr, reduced_tr), axis=1)
                    threads = []
                    for jj in range(cv_iteration):
                        X = _dataset[:, 0:-1]
                        y = _dataset[:, -1]
                        threads = threads + [
                            Process(target=self.worker_thread,
                                    args=(clf, X, y, k, jj, precision, recall, f1, accuracy, mcc, auc))]
                    for ii in range(cv_iteration):
                        threads[ii].start()
                    for ii in range(cv_iteration):
                        threads[ii].join()
    return precision, recall, f1, accuracy, mcc, auc


def main():
    config_indicator = 1
    ch_obj = LoadConfig(config_indicator)
    configuration = ch_obj.exp_configs

    dataset_names, dataset, datasets_file_names = dh_obj.load_datasets(configuration, drop_unused_columns="new")

    bench_obj = Benchmarks(dataset, dataset_names, datasets_file_names, configuration)
    if configuration['validation_type'] == 0:
        bench_obj.differen_release_MLMNB()
    if configuration['validation_type'] == 1:
        bench_obj.different_release()
    if configuration['validation_type'] == 2:
        t1 = time.time()
        bench_obj.cross_validation()
        t2 = time.time()
        print("time:", t2 - t1)
        # precision, recall, f1, accuracy, mcc, auc = bench_obj.cv_parallel(10)


if __name__ == '__main__':
    main()
