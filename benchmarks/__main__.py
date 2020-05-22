import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, \
    roc_auc_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from data_collection_manipulation.data_handler import IO
from configuration_files.setup_config import LoadConfig
from data_collection_manipulation.data_handler import DataPreprocessing
import pandas as pd

gnb_obj = GaussianNB()
rnd_obj = RandomForestClassifier()
dh_obj = IO()


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

    def compute_measures(self, X_train, X_test, actual, pred, s_attrib):
        perf_pack = []
        a = classification_report(actual, pred, output_dict=True)
        if self.precision_flag:
            perf_pack.append(round(a['2.0']['precision'], 2))
        if self.recall_flag:
            perf_pack.append(round(a['2.0']['recall'], 2))
        if self.F1_flag:
            # f1 = 2 * a['2.0']['precision'] * a['2.0']['recall'] / (a['2.0']['precision'] + a['2.0']['recall'])
            perf_pack.append(round(a['2.0']['f1-score'], 2))
        if self.ACC_flag:
            ACC = accuracy_score(actual, pred)
            perf_pack.append(round(ACC, 2))
        if self.MCC_flag:
            MCC = matthews_corrcoef(actual, pred, sample_weight=None)
            perf_pack.append(round(MCC, 2))
        if self.AUC_flag:
            _auc = roc_auc_score(actual, pred, average=None)
            perf_pack.append(round(roc_auc_score(actual, pred, average=None), 2))

        # conf_mat = confusion_matrix(actual, pred, labels=range(s_attrib[-1]))
        #
        # num_incorrect = ((actual != pred).sum())
        # num_correct = ((actual == pred).sum())
        # num_training_instances = X_train.shape[0]
        # num_test_instances = X_test.shape[0]
        # percent_incorrect = (num_incorrect / num_test_instances)

        # stat = [num_training_instances, num_test_instances, num_correct, num_incorrect, percent_incorrect]
        # perf_pack.append(stat)
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
            self.validator = StratifiedKFold()
        elif self.config['cross_validation_type'] == 3:
            self.validator = KFold(n_splits=self.config['number_of_folds'])

        self.classifiers = [
            RandomForestClassifier(n_estimators=1, bootstrap=False, n_jobs=-1, max_depth=3, max_features=None,
                                   random_state=99),
            GaussianNB(), LogisticRegression()]

        self.header2 = ["Defect models", "projects' category", "Tr version", "Ts version", "Iterations",
                        "Precision", "Recall", "F1", "ACC", "MCC", "AUC"]

        self.header1 = ["File name", "original status", "predicted status"]

        self.perf_obj = PerformanceEvaluation(self.config)

        self.temp_addr = "E:\\apply\\york\\project\\source\\outputs\\file_level" \
                         "\\different_releases_tr_ts\\res_jmt.csv"

    def different_release(self):
        self.dataset = DataPreprocessing.binerize_class(self.dataset)
        temp_result = [self.header2]
        for model_name, clf in zip(self.model_holder, self.classifiers):
            for ds_cat, ds_val in self.dataset.items():
                for i, ds_version in enumerate(ds_val):
                    for j in range(i + 1, len(ds_val)):
                        ## think about attrib size for different releases.
                        ds_val[i][:] = np.nan_to_num(ds_val[i])
                        ds_val[j][:] = np.nan_to_num(ds_val[j])
                        self.s_attrib = DataPreprocessing.get_metrics_size(data=ds_val[j])
                        tr = np.array(ds_val[i])
                        ts = np.array(ds_val[j])
                        X_train = tr[:, 0:-2]
                        y_train = tr[:, -1]
                        X_test = ts[:, 0:-2]
                        y_test = ts[:, -1]
                        for iterations in range(self.config['iterations']):
                            clf.fit(X_train, y_train)
                            random.seed(100)
                            y_pred = clf.predict(X_test)

                            print(self.dataset_names[ds_cat][i])
                            perf_holder = self.perf_obj.compute_measures(X_train, X_test, y_test, y_pred,
                                                                         self.s_attrib)

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

    def cross_validation(self):
        temp_result = []
        for model_name, clf in zip(self.model_holder, self.classifiers):
            for key_dataset in range(len(self.dataset)):
                _dataset = np.array(self.dataset[key_dataset])
                for key_iter in range(self.config['iterations']):
                    X = _dataset[:, 0:-1]
                    y = _dataset[:, -1]

                    metric_sizes = DataPreprocessing.get_metrics_size(_dataset)

                    class_probability_holder = np.zeros((metric_sizes[-1], 1))

                    k = 0

                    for train_idx, test_idx in self.validator.split(X, y):
                        X_train, X_test = X[train_idx], X[test_idx]
                        # y_test = actual class label of test data
                        # y_train = actual class label of train data
                        y_train, y_test = y[train_idx], y[test_idx]

                        predicted = []

                        clf.fit(X_train, y_train)
                        score = clf.predict(X_test)
                        prob = clf.predict_proba(X_test)

                        perf_holder = self.perf_obj.compute_measures(y_test, score)

                        cross_val_pack = [str(self.dataset_names[key_dataset]), key_iter, k, model_name, perf_holder]

                        k = k + 1

                        temp_result.append(cross_val_pack)
        dh_obj.write_csv(temp_result, self.config['file_level_WPDP_cross_validation_results_des'])


def main():
    config_indicator = 1
    ch_obj = LoadConfig(config_indicator)
    configuration = ch_obj.exp_configs

    dataset_names, dataset, datasets_file_names = dh_obj.load_datasets(configuration, drop_unused_columns="new")

    bench_obj = Benchmarks(dataset, dataset_names, datasets_file_names, configuration)

    if configuration['validation_type'] == 1:
        bench_obj.different_release()
    if configuration['validation_type'] == 2:
        bench_obj.cross_validation()


if __name__ == '__main__':
    main()
