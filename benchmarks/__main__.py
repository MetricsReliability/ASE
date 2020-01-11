import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from data_collection_manipulation.data_handler import IO
from configuration_files.setup_config import LoadConfig
from data_collection_manipulation.data_handler import DataPreprocessing

gnb_obj = GaussianNB()
rnd_obj = RandomForestClassifier()
dh_obj = IO()


class PerformanceEvaluation:
    def __init__(self, configuration):
        self.config = configuration
        self.recall_flag = False
        self.precision_flag = False
        self.F1 = False
        self.ACC_flag = False
        self.MCC_flag = False

        for measure in self.config['evaluation_measures']:
            if measure == 'Precision':
                self.precision_flag = True
            if measure == 'Recall':
                self.recall_flag = True
            if measure == 'ACC':
                self.ACC_flag = True
            if measure == 'MCC':
                self.MCC_flag = True
            if measure == 'F1':
                self.F1 = True

    def compute_measures(self, actual, pred):
        perf_pack = []
        if self.ACC_flag:
            ACC = accuracy_score(actual, pred)
            perf_pack.append(ACC)
        if self.F1:
            _F1 = f1_score(actual, pred, average='weighted')
            perf_pack.append(_F1)
        if self.precision_flag:
            Prec = precision_score(actual, pred, labels=None, average='weighted')
            perf_pack.append(Prec)
        if self.recall_flag:
            Reca = recall_score(actual, pred, labels=None, average='weighted')
            perf_pack.append(Reca)
        if self.MCC_flag:
            MCC = matthews_corrcoef(actual, pred, sample_weight=None)
            perf_pack.append(MCC)

        return perf_pack

    @staticmethod
    def serializer(key_dataset, key_run, key_fold, key_scheme, perf_holder):
        data = [key_dataset, key_run, key_fold, key_scheme, *perf_holder]
        return data


class Benchmarks:
    def __init__(self, dataset, dataset_names, _configuration):
        self.dataset = dataset
        self.config = _configuration
        self.dataset_names = dataset_names
        self.model_holder = self.config['defect_models']

        if self.config['cross_validation_type'] == 1:
            self.validator = LeaveOneOut()
        if self.config['cross_validation_type'] == 2:
            self.validator = StratifiedKFold()
        if self.config['cross_validation_type'] == 3:
            self.validator = KFold(n_splits=self.config['number_of_folds'])

        self.classifiers = [
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            GaussianNB()]

    def base_experimenter(self):
        perf_obj = PerformanceEvaluation(self.config)
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
                        y_train, y_test = y[train_idx], y[test_idx]

                        predicted = []

                        clf.fit(X_train, y_train)
                        score = clf.predict(X_test)
                        prob = clf.predict_proba(X_test)

                        perf_holder = perf_obj.compute_measures(y_test, score)

                        serialized_data = PerformanceEvaluation.serializer(
                            str(self.dataset_names[key_dataset]), key_iter, k, model_name, perf_holder)
                        k = k + 1

                        temp_result.append(serialized_data)
        dh_obj.write_csv(temp_result, self.config['results_destination'])


def main():
    config_indicator = 1
    ch_obj = LoadConfig(config_indicator)
    configuration = ch_obj.exp_configs

    dataset, dataset_names = dh_obj.load_datasets(configuration)

    bench_obj = Benchmarks(dataset, dataset_names, configuration)
    bench_obj.base_experimenter()

    # # print("Number of mislabeled points out of a total %d points : %d" %
    # #       (X_test.shape[0], (y_test != y_pred).sum()))
    # # class_prob = mnb_obj.fit(X_train, y_train).predict_proba(X_test)
    # scores = cross_val_score(mnb_obj, X, y, cv=10, scoring='f1_macro')
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == '__main__':
    main()
