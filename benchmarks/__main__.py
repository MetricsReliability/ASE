import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from data_collection_manipulation.data_handler import IO
from configuration_files.setup_config import LoadConfig
from data_collection_manipulation.data_handler import DataPreprocessing
import operator

mnb_obj = MultinomialNB()
rnd_obj = RandomForestClassifier()


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
        if self.ACC_flag:
            ACC = accuracy_score(actual, pred)
        if self.F1:
            _F1 = f1_score(actual, pred)
        if self.precision_flag:
            Prec = precision_score(actual, pred, labels=None, average='weighted')
        if self.recall_flag:
            Reca = recall_score(actual, pred, labels=None, average='weighted')
        if self.MCC_flag:
            MCC = matthews_corrcoef(actual, pred, sample_weight=None)

        perf_pack = ACC, Prec, Reca, MCC
        return perf_pack

    @staticmethod
    def serializer(self, key_dataset, key_run, key_fold, key_scheme, perf_holder):
        data = [key_dataset, key_run, key_fold, key_scheme, perf_holder[0],
                perf_holder[1], perf_holder[2], perf_holder[3]]
        return data


class Benchmarks:
    def __init__(self, dataset, dataset_names, _configuration):
        self.dataset = dataset
        self.config = _configuration
        self.dataset_names = dataset_names

        if self.config['experiment_mode'] == 1 and self.config['iterations'] > 1:
            raise Exception("The iterations parameter with a value more than zero is for the experimenter mode! ")
        if self.config['experiment_mode'] == 2 and len(self.config['defect_models']) <= 0:
            raise Exception("Please specify a defect model for prediction!")

        self.model_holder = []
        for item in self.config['defect_models']:
            if item == "Mnb":
                self.model_holder.append(mnb_obj)
            if item == "RndFor":
                self.model_holder.append(rnd_obj)

    def leave_one_out(self):
        perf_obj = PerformanceEvaluation(self.config)
        for key_model in range(len(self.model_holder)):
            for key_dataset in range(len(self.dataset)):
                _dataset = np.array(self.dataset[key_dataset])
                for key_iter in range(self.config['iterations']):
                    i = 0
                    [m, n] = np.shape(_dataset)
                    metric_sizes = DataPreprocessing.get_metrics_size(_dataset)
                    class_probability_holder = np.zeros((metric_sizes[-1], 1))

                    predicted = []
                    while i < m:
                        test_data = _dataset[i, 0:-1].reshape(1, -1)
                        test_data_reserve = _dataset[0, :]
                        _dataset = np.delete(_dataset, np.s_[0], axis=0)

                        _pred_prob = mnb_obj.fit(_dataset[:, 0:-1], _dataset[:, -1]).predict_proba(test_data)

                        class_probability_holder = np.concatenate((class_probability_holder, _pred_prob.T), axis=1)
                        _dataset = np.concatenate((_dataset, test_data_reserve.reshape(1, -1)), axis=0)

                        index, pred = max(enumerate(_pred_prob), key=operator.itemgetter(1))
                        pred = pred.max()
                        predicted.append(int(pred))
                        i = i + 1
                    class_probability_holder = np.delete(class_probability_holder, 0, 1)

                    perf_holder = perf_obj.compute_measures(_dataset[:, -1], predicted)

                    serialized_data = PerformanceEvaluation.serializer(
                        str(self.dataset_names[key_dataset]), key_iter, i, 'MNB', perf_holder)


def main():
    config_indicator = 1
    ch_obj = LoadConfig(config_indicator)
    configuration = ch_obj.exp_configs

    dh_obj = IO()
    dataset, dataset_names = dh_obj.load_datasets(configuration)

    bench_obj = Benchmarks(dataset, dataset_names, configuration)
    bench_obj.leave_one_out()

    # # print("Number of mislabeled points out of a total %d points : %d" %
    # #       (X_test.shape[0], (y_test != y_pred).sum()))
    # # class_prob = mnb_obj.fit(X_train, y_train).predict_proba(X_test)
    # scores = cross_val_score(mnb_obj, X, y, cv=10, scoring='f1_macro')
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == '__main__':
    main()
