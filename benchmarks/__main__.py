import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, KFold
from data_collection_manipulation.data_handler import IO
from configuration_files.setup_config import LoadConfig
from unittest import TestCase

dh_obj = IO()
mnb_obj = MultinomialNB()

config_indicator = 1
ch_obj = LoadConfig(config_indicator)
configuration = ch_obj.exp_configs


class Benchmarks:
    def __init__(self, dataset, _configuration):
        self.dataset = dataset
        self.config = _configuration
        if self.config['experiment_mode'] == 1 and self.config['iterations'] > 1:
            raise Exception("The iterations parameter with a value more than zero is for the experimenter mode! ")
        if self.config['experiment_mode'] == 2 and len(self.config['defect_models']) <= 0:
            raise Exception("Please specify a defect model for prediction!")

    def leave_one_out(self, dataset_pack):
        i = 0
        while i <= self.sample_size:
            # test_instance = trdata.iloc[i, :]
            # trdata.drop([i], inplace=True)
            # trdata.reset_index(drop=True, inplace=True)

            test_data = data[i, 0:-1].reshape(1, -1)
            data = np.delete(data, np.s_[i], axis=0)
            class_prob = mnb_obj.fit(data[:, 0:-1], data[:, -1]).predict_proba(test_data)

            data = np.concatenate((data,), axis=0)
            return self.dataset

    def check_ndarray(self, flag):
        dataset_pack = []
        if flag == 1:
            for _dataset in self.dataset:
                dataset_pack.append(np.array(_dataset))
        return dataset_pack

    def exec(self):
        ndarray_flag = 0
        if self.config['dataset_first']:
            temp_1 = self.dataset
            temp_2 = self.config['defect_models']
            ndarray_flag = 1
        if not self.config['dataset_first']:
            temp_1 = self.config['defect_models']
            temp_2 = self.dataset
        dataset_pack = self.check_ndarray(ndarray_flag)

        for item_1 in range(len(temp_1)):
            if ndarray_flag:
                _dataset = dataset_pack[item_1]
            for key_run in range(self.config['iterations']):
                for item_2 in range(len(temp_2)):
                    self.leave_one_out(dataset_pack)


def main():
    dataset = dh_obj.load_datasets(configuration)

    bench_obj = Benchmarks(dataset, configuration)
    bench_obj.exec()

    # # print("Number of mislabeled points out of a total %d points : %d" %
    # #       (X_test.shape[0], (y_test != y_pred).sum()))
    # # class_prob = mnb_obj.fit(X_train, y_train).predict_proba(X_test)
    # scores = cross_val_score(mnb_obj, X, y, cv=10, scoring='f1_macro')
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == '__main__':
    main()
