import random
import time
from multiprocessing import Process, Manager
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from libs.load_CPDP import load_CPDP_datasets
from sklearn.utils import shuffle
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, \
    roc_auc_score, classification_report
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from data_collection_manipulation.data_handler import IO
from configuration_files.setup_config import LoadConfig
from data_collection_manipulation.data_handler import DataPreprocessing
from sklearn import tree
import pandas as pd
from keras import backend as b
from libs.MLMNB_package_python import MLMNB
from sklearn.preprocessing import KBinsDiscretizer
from libs.feature_selection import feature_selection
# from libs.TCA import TCA
# from tl_algs import tl_alg
from da_tool.tca import TCA

from libs import vae

dh_obj = IO()
disc = KBinsDiscretizer(n_bins=10, encode='ordinal')
from latent_base import AmlmnbBase
from latents import AMLMNB


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
            perf_pack.append(round(a['2.0']['precision'], 2))
        if self.recall_flag:
            perf_pack.append(round(a['2.0']['recall'], 2))
        if self.F1_flag:
            perf_pack.append(round(a['2.0']['f1-score'], 2))
        if self.ACC_flag:
            ACC = accuracy_score(actual, pred)
            perf_pack.append(round(ACC, 2))
        if self.MCC_flag:
            MCC = matthews_corrcoef(actual, pred, sample_weight=None)
            perf_pack.append(round(MCC, 2))
        if self.AUC_flag:
            perf_pack.append(round(roc_auc_score(actual, pred, average=None), 2))
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
        self.nb_epoch = 100
        self.batch_size = 50
        self.encoding_dim = 32
        self.learning_rate = 1e-3

        self.classifiers = [RandomForestClassifier(), MultinomialNB(), LogisticRegression(),
                            tree.DecisionTreeClassifier(), AdaBoostClassifier(n_estimators=100),
                            KNeighborsClassifier(n_neighbors=3)]

        # self.classifiers = []

        self.header2 = ["Defect models", "projects' category", "Tr version", "Ts version", "Iterations",
                        "Precision", "Recall", "F1 Score", "ACC", "MCC", "AUC"]

        self.header3 = ["Defect models", "pair", "Iterations",
                        "Precision", "Recall", "F1 Score", "ACC", "MCC", "AUC"]

        self.header1 = ["File name", "original status", "predicted status"]

        self.cv_header = ["Key_Dataset", "Key_Run", "Key_Fold", "Key_Scheme", "Precsion", "Recall", "F1_Score", "ACC",
                          "MCC", "AUC"]

        self.perf_obj = PerformanceEvaluation(self.config)

        self.temp_addr = "E:\\apply\\york\\project\\source\\outputs\\file_level" \
                         "\\different_releases_tr_ts\\res_jmt.csv"

    def remove_leading_zero(self, list):
        rec = []
        for item in list:
            a = np.format_float_positional(item, trim='-')
            if a != str(item):
                a = int(a)
            else:
                a = float(a)
            rec.append(isinstance(a, float))
        return rec

    def wpdp(self):
        children = []
        learner = AmlmnbBase(h=[], h_states=[1], delta=0.001, alpha=0.00001, mode_h='individual11')
        au = vae.AutoEncoder(self.nb_epoch, self.batch_size, self.encoding_dim, self.learning_rate)
        self.classifiers.append(learner)
        self.classifiers.append(au)
        temp = self.classifiers[0]
        self.classifiers[0] = learner
        self.classifiers[-2] = temp

        # MNB_CLF = AmlmnbBase(h=[], h_states=[1], delta=0.001, alpha=0.00001, mode_h='individual')
        # self.model_holder[1] = "MNB"
        # self.classifiers[1] = MNB_CLF

        self.dataset = DataPreprocessing.binerize_class(self.dataset)
        temp_result = [self.header2]
        for model_name, clf in zip(self.model_holder, self.classifiers):
            for ds_cat, ds_val in self.dataset.items():
                for i in range(len(ds_val)):
                    for j in range(i + 1, len(ds_val)):
                        tr = np.array(ds_val[i])
                        ts = np.array(ds_val[j])

                        if model_name == "MLMNB":
                            ss = [[], []]
                            for counter in range(np.size(tr, 1) - 1):
                                ss[0].append(sum(tr[:, counter]))
                                ss[1].append(counter)
                            c = tr[:, -1]
                            tr = SelectKBest(score_func=mutual_info_classif,
                                             k=round(np.size(tr, 1) / 2)).fit_transform(tr[:, 0:-1],
                                                                                        tr[:, -1])

                            tr = np.concatenate((tr, c.reshape(-1, 1)), axis=1)
                            idxx = []
                            for counter1 in range(len(ss[0])):
                                for counter2 in range(np.size(tr, 1) - 1):
                                    if ss[0][counter1] == sum(tr[:, counter2]):
                                        idxx.append(ss[1][counter1])
                            cts = ts[:, -1]
                            ts = ts[:, idxx]
                            ts = np.concatenate((ts, cts.reshape(-1, 1)), axis=1)

                            array_types = [sum(tr[:, feature]) for feature in
                                           range(np.size(tr, 1) - 1)]
                            array_types = self.remove_leading_zero(array_types)
                            conti_features = []
                            for idx, e in enumerate(array_types):
                                if e == True:
                                    conti_features.append(idx)
                            for k in range(np.size(tr, 1) - 1):
                                children.append(k)
                            learner = AmlmnbBase(h=[conti_features], h_states=[5], delta=0.01, alpha=0.001,
                                                 mode_h='individual')
                            clf = learner
                            # if ds_cat == "CM1" or ds_cat == "JM1" or ds_cat == "KC1" or ds_cat == "KC2" or ds_cat == "KC3":
                            #     tr = self.delete_unused_NASA_metrics(tr, ds_cat, conti_features)
                            #     ts = self.delete_unused_NASA_metrics(ts, ds_cat, conti_features)
                            # else:
                            # _dataset[:, 0:-1] = disc.fit_transform(_dataset[:, 0:-1])
                            # reduced_tr = np.delete(tr, conti_features, axis=1)
                            # discretized_tr = disc.fit_transform(tr[:, conti_features])
                            # tr = np.concatenate((discretized_tr, reduced_tr), axis=1)
                            #
                            # reduced_ts = np.delete(ts, conti_features, axis=1)
                            # discretized_ts = disc.fit_transform(ts[:, conti_features])
                            # ts = np.concatenate((discretized_ts, reduced_ts), axis=1)

                        X_train = tr[:, 0:-1]
                        y_train = tr[:, -1]
                        X_test = ts[:, 0:-1]
                        y_test = ts[:, -1]

                        for iterations in range(self.config['iterations']):
                            if model_name == "DNN":
                                df_train = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
                                df_test = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)

                                df_train_1 = df_train[df_train[:, -1] == 1]
                                df_train_2 = df_train[df_train[:, -1] == 2]
                                df_train_1_x = np.delete(df_train_1, -1, axis=1)
                                # df_train_2_x = np.delete(df_train_2, -1, axis=1)

                                df_test_1 = df_test[df_test[:, -1] == 1]
                                df_test_2 = df_test[df_test[:, -1] == 2]
                                df_test_1_x = np.delete(df_test_1, -1, axis=1)
                                df_test_2_x = np.delete(df_test_2, -1, axis=1)

                                b.clear_session()
                                clf.fit(df_train_1_x)
                                perf_holder = clf.predict(df_test)
                            else:
                                clf.fit(X_train, y_train)
                                random.seed(100)
                                y_pred = clf.predict(X_test)

                                y_pred = np.array(y_pred)
                                perf_holder = self.perf_obj.compute_measures(y_test, y_pred)

                            print(self.dataset_names[ds_cat][i])

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

            dh_obj.write_csv(temp_result, self.config['file_level_different_release_results_whole'])

    def cpdp(self):
        children = []
        learner = AmlmnbBase(h=[], h_states=[1], delta=0.001, alpha=0.00001, mode_h='individual11')
        au = vae.AutoEncoder(self.nb_epoch, self.batch_size, self.encoding_dim, self.learning_rate)
        self.classifiers.append(learner)
        self.classifiers.append(au)
        temp = self.classifiers[0]
        self.classifiers[0] = learner
        self.classifiers[-2] = temp

        data_pairs = load_CPDP_datasets()
        data_pairs = DataPreprocessing.binerizeCPDP(data_pairs)
        temp_result = [self.header3]
        for model_name, clf in zip(self.model_holder, self.classifiers):
            for pair, data in enumerate(data_pairs):
                tr = np.asarray(data[0])
                ts = np.asarray(data[1])

                if model_name == "MLMNB":
                    # tca_obj = TCA(dim=tr.shape[1] - 1, kerneltype='linear', kernelparam=0.1, mu=1)
                    # tr[:, 0:-1], ts[:, 0:-1] = tca_obj.fit_transform(tr[:, 0:-1], ts[:, 0:-1])

                    #########################################################
                    ss = [[], []]
                    for i in range(np.size(tr, 1) - 1):
                        ss[0].append(sum(tr[:, i]))
                        ss[1].append(i)
                    c = tr[:, -1]
                    tr = SelectKBest(score_func=mutual_info_classif,
                                     k=round(np.size(tr, 1) / 2)).fit_transform(tr[:, 0:-1],
                                                                                tr[:, -1])

                    tr = np.concatenate((tr, c.reshape(-1, 1)), axis=1)
                    idxx = []
                    for i in range(len(ss[0])):
                        for j in range(np.size(tr, 1) - 1):
                            if ss[0][i] == sum(tr[:, j]):
                                idxx.append(ss[1][i])
                    cts = ts[:, -1]
                    ts = ts[:, idxx]
                    ts = np.concatenate((ts, cts.reshape(-1, 1)), axis=1)

                    #####################################################
                    array_types = [sum(tr[:, feature]) for feature in
                                   range(np.size(tr, 1) - 1)]
                    array_types = self.remove_leading_zero(array_types)
                    conti_features = []
                    for idx, e in enumerate(array_types):
                        if e == True:
                            conti_features.append(idx)
                    ######################################################
                    for k in range(np.size(tr, 1) - 1):
                        children.append(k)
                    learner = AmlmnbBase(h=[children], h_states=[5], delta=0.01, alpha=0.001,
                                         mode_h='individual')
                    clf = learner
                    ##########################################################

                    reduced_tr = np.delete(tr, conti_features, axis=1)
                    discretized_tr = disc.fit_transform(tr[:, conti_features])
                    tr = np.concatenate((discretized_tr, reduced_tr), axis=1)

                    reduced_ts = np.delete(ts, conti_features, axis=1)
                    discretized_ts = disc.fit_transform(ts[:, conti_features])
                    ts = np.concatenate((discretized_ts, reduced_ts), axis=1)

                X_train = tr[:, 0:-1]
                y_train = tr[:, -1]
                X_test = ts[:, 0:-1]
                y_test = ts[:, -1]

                for iterations in range(self.config['iterations']):
                    print("MODEL:", model_name, "PAIR:", pair, "ITERATION:", iterations)
                    if model_name == "DNN":
                        df_train = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
                        df_test = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)

                        df_train_1 = df_train[df_train[:, -1] == 1]
                        df_train_2 = df_train[df_train[:, -1] == 2]
                        df_train_1_x = np.delete(df_train_1, -1, axis=1)
                        # df_train_2_x = np.delete(df_train_2, -1, axis=1)

                        df_test_1 = df_test[df_test[:, -1] == 1]
                        df_test_2 = df_test[df_test[:, -1] == 2]
                        df_test_1_x = np.delete(df_test_1, -1, axis=1)
                        df_test_2_x = np.delete(df_test_2, -1, axis=1)

                        b.clear_session()
                        clf.fit(df_train_1_x)
                        perf_holder = clf.predict(df_test)
                    else:
                        clf.fit(X_train, y_train)
                        random.seed(100)
                        y_pred = clf.predict(X_test)

                        y_pred = np.array(y_pred)
                        perf_holder = self.perf_obj.compute_measures(y_test, y_pred)

                    release_pack = [model_name, pair, iterations, *perf_holder]

                    temp_result.append(release_pack)

                    dh_obj.write_csv(temp_result, self.config['file_level_different_release_results_whole'])
        return self

    def delete_unused_NASA_metrics(self, _dataset, ds_cat, conti_features):
        if ds_cat == "CM1":
            # [7, 9, 11, 14, 17, 18, 19, 20, 22, 23, 24, 25, 29, 35]
            reduced_dataset = np.delete(_dataset, conti_features, axis=1)
            discretized_tr = disc.fit_transform(_dataset[:, conti_features])
            _dataset = np.concatenate((discretized_tr, reduced_dataset), axis=1)
        if ds_cat == "JM1":
            # [8, 9, 10, 11, 13, 14, 15]
            reduced_dataset = np.delete(_dataset, conti_features, axis=1)
            discretized_tr = disc.fit_transform(_dataset[:, conti_features])
            _dataset = np.concatenate((discretized_tr, reduced_dataset), axis=1)
        if ds_cat == "KC1" or ds_cat == "KC2" or ds_cat == "KC3":
            reduced_dataset = np.delete(_dataset, conti_features, axis=1)
            discretized_tr = disc.fit_transform(_dataset[:, conti_features])
            _dataset = np.concatenate((discretized_tr, reduced_dataset), axis=1)
        return _dataset

    def cross_validation(self):
        temp_result = [self.cv_header]
        children = []
        self.model_holder.append("MLMNB")
        children = []
        learner = AmlmnbBase(h=[], h_states=[1], delta=0.001, alpha=0.00001, mode_h='individual11')
        self.classifiers.append(learner)
        temp = self.classifiers[0]
        self.classifiers[0] = learner
        self.classifiers[-1] = temp

        self.dataset = DataPreprocessing.binerize_class(self.dataset)
        for model_name, clf in zip(self.model_holder, self.classifiers):
            if model_name == "MLMNB":
                for ii in range(1, 10):
                    for ds_cat, ds_val in self.dataset.items():
                        for i in range(len(ds_val)):
                            _dataset = np.array(ds_val[i])
                            c = _dataset[:, -1]
                            _dataset = SelectKBest(score_func=mutual_info_classif,
                                                   k=round(np.size(_dataset, 1) / 2)).fit_transform(_dataset[:, 0:-1],
                                                                                                    _dataset[:, -1])
                            _dataset = np.concatenate((_dataset, c.reshape(-1, 1)), axis=1)

                            array_types = [isinstance(sum(_dataset[:, feature]), int) for feature in
                                           range(np.size(_dataset, 1) - 1)]
                            conti_features = []
                            for idx, e in enumerate(array_types):
                                if e == False:
                                    conti_features.append(idx)
                            for k in range(np.size(_dataset, 1) - 1):
                                children.append(k)
                            if ii == 1:
                                learner = AmlmnbBase(h=[], h_states=[ii], delta=0.001, alpha=0.00001)
                            else:
                                learner = AmlmnbBase(h=[children], h_states=[ii], delta=0.001, alpha=0.00001)
                            clf = learner
                            # if ds_cat == "CM1" or ds_cat == "JM1" or ds_cat == "KC1" or ds_cat == "KC2" or ds_cat == "KC3":
                            #     _dataset = self.delete_unused_NASA_metrics(_dataset, ds_cat, conti_features)
                            # else:
                            #     reduced_tr = np.delete(_dataset, conti_features, axis=1)
                            #     discretized_tr = disc.fit_transform(_dataset[:, conti_features])
                            #     _dataset = np.concatenate((discretized_tr, reduced_tr), axis=1)

                            for key_iter in range(self.config['iterations']):
                                X = _dataset[:, 0:-1]
                                y = _dataset[:, -1]
                                k = 0
                                for train_idx, test_idx in self.validator.split(X, y):
                                    print('CLASSIFIER:', model_name + str(ii), "DATASET", self.dataset_names[ds_cat][i],
                                          'ITERATION:',
                                          key_iter, 'CV_FOLD:', k)
                                    X_train, X_test = X[train_idx], X[test_idx]
                                    # y_test = actual class label of test data
                                    # y_train = actual class label of train data
                                    y_train, y_test = y[train_idx], y[test_idx]
                                    clf.fit(X_train, y_train)
                                    score = clf.predict(X_test)
                                    perf_holder = self.perf_obj.compute_measures(y_test, score)
                                    cross_val_pack = [str(self.dataset_names[ds_cat][i]), key_iter, k,
                                                      model_name + str(ii),
                                                      *perf_holder]

                                    k = k + 1

                                    temp_result.append(cross_val_pack)
            else:
                for ds_cat, ds_val in self.dataset.items():
                    for i in range(len(ds_val)):
                        _dataset = np.array(ds_val[i])
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

                                clf.fit(X_train, y_train)
                                score = clf.predict(X_test)
                                perf_holder = self.perf_obj.compute_measures(y_test, score)
                                cross_val_pack = [str(self.dataset_names[ds_cat][i]), key_iter, k, model_name,
                                                  *perf_holder]

                                k = k + 1

                                temp_result.append(cross_val_pack)

            dh_obj.write_csv(temp_result, self.config['file_level_WPDP_cross_validation_results_des'])

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
        bench_obj.cv_parallel(10)
    if configuration['validation_type'] == 1:
        bench_obj.wpdp()
    if configuration['validation_type'] == 2:
        t1 = time.time()
        bench_obj.cross_validation()
        t2 = time.time()
        print("time:", t2 - t1)
        # precision, recall, f1, accuracy, mcc, auc = bench_obj.cv_parallel(10)
    if configuration['validation_type'] == 3:
        bench_obj.cpdp()


if __name__ == '__main__':
    main()
