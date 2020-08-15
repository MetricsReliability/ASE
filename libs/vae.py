from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Dense, regularizers
from keras.models import Model
from keras.datasets import mnist
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import precision_recall_curve, classification_report, roc_auc_score, accuracy_score, \
    matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from numpy.random import seed
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
from genetic_selection import GeneticSelectionCV
from sklearn.preprocessing import StandardScaler

from configuration_files.setup_config import LoadConfig
from libs.feature_selection import selection_sort
from data_collection_manipulation.data_handler import DataPreprocessing, IO
from benchmarks.__main__ import PerformanceEvaluation
import data_collection_manipulation.data_handler

validator = StratifiedKFold(n_splits=10, random_state=2, shuffle=True)

dh_obj = IO()

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)
from keras import backend as b

def binerizeCPDP(data):
    for r in range(len(data)):
        if data.iloc[r, -1] > 0:
            data.iloc[r, -1] = 2
        else:
            data.iloc[r, -1] = 1
    return data

class AutoEncoder:
    def __init__(self, epoch, batch_size, encoding_dim, learning_rate):
        self.nb_epoch = epoch
        self.batch_size = batch_size
        self.encoding_dim = encoding_dim
        self.hidden_dim = int(encoding_dim / 2)
        self.learning_rate = learning_rate

    def fit(self, df_train_1_x):
        input_dim = df_train_1_x.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(128, activation='sigmoid')(input_layer)
        encoded = Dense(64, activation='sigmoid')(encoded)
        encoded = Dense(32, activation='sigmoid')(encoded)

        decoded = Dense(32, activation='sigmoid')(encoded)
        decoded = Dense(64, activation='sigmoid')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)

        self.autoencoder = Model(inputs=input_layer, outputs=decoded)
        self.autoencoder.compile(metrics=['accuracy'],
                                 loss='mean_squared_error',
                                 optimizer='adam')

        self.autoencoder.fit(df_train_1_x, df_train_1_x,
                             epochs=self.nb_epoch,
                             batch_size=self.batch_size,
                             shuffle=True,
                             # validation_data=(df_valid_1_x, df_valid_1_x),
                             verbose=1,
                             )

    def predict(self, df_test):
        test_x_predictions = self.autoencoder.predict(df_test[:, 0:-1])

        mse = np.mean(np.power(df_test[:, 0:-1] - test_x_predictions, 2), axis=1)
        error_df_test = pd.DataFrame({'Reconstruction_error': mse,
                                      'True_class': df_test[:, -1]})
        error_df_test = error_df_test.reset_index()

        precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df_test.True_class,
                                                                       error_df_test.Reconstruction_error, pos_label=2)

        store = np.zeros((6, len(threshold_rt)))

        a = []
        idx = []
        for i in range(len(threshold_rt)):
            threshold_fixed = threshold_rt[i]

            pred_y = [2 if e > threshold_fixed else 1 for e in error_df_test.Reconstruction_error.values]

            report = classification_report(error_df_test.True_class, pred_y, output_dict=True)

            # store[0, i] = threshold_fixed
            store[0, i] = round(report['2.0']['precision'], 2)
            store[1, i] = round(report['2.0']['recall'], 2)
            store[2, i] = round(report['2.0']['f1-score'], 2)
            store[3, i] = round(accuracy_score(error_df_test.True_class, pred_y), 2)
            store[4, i] = round(matthews_corrcoef(error_df_test.True_class, pred_y), 2)
            store[5, i] = round(roc_auc_score(error_df_test.True_class, pred_y, average=None), 2)

            a.append(sum(store[:, i]))
            idx.append(i)
        #
        # for i in range(store.shape[1]):
        #     a.append(sum(store[:, i]))
        #
        idx = np.argsort(a)
        # [_, idx] = selection_sort(a, idx)
        prec = store[:, idx[-1]][0]
        reca = store[:, idx[-1]][1]
        f1 = store[:, idx[-1]][2]
        acc = store[:, idx[-1]][3]
        mcc = store[:, idx[-1]][4]
        auc = store[:, idx[-1]][5]

        return [prec, reca, f1, acc, mcc, auc]


def main():
    SEED = 123
    DATA_SPLIT_PCT = 0.25
    config_indicator = 1
    ch_obj = LoadConfig(config_indicator)
    configuration = ch_obj.exp_configs
    nb_epoch = 100
    batch_size = 50
    encoding_dim = 32
    learning_rate = 1e-3
    au = AutoEncoder(nb_epoch, batch_size, encoding_dim, learning_rate)
    dataset_names, dataset, datasets_file_names = dh_obj.load_datasets(configuration, drop_unused_columns="new")
    dataset = DataPreprocessing.binerize_class(dataset)

    temp_result = [["Key_Dataset", "Key_Run", "Key_Fold", "Key_Scheme", "Precsion", "Recall", "F1_Score", "ACC",
                    "MCC", "AUC"]]
    for ds_cat, ds_val in dataset.items():
        for i in range(len(ds_val)):
            _dataset = np.array(ds_val[i])
            if ds_cat == "CM1":
                _dataset = _dataset[:, [0, 1, 3, 10, 13, 31, 6, 18, 19, 17, 20, 21, 22, 32, 33, -1]]
            elif ds_cat == "JM1" or ds_cat == "KC1" or ds_cat == "KC2" or ds_cat == "PC1":
                _dataset = _dataset[:, [0, 1, 2, 3, 6, 7, 8, 9, 10, 15, 16, 17, 18, 20, -1]]

            X = _dataset[:, 0:-1]
            y = _dataset[:, -1]
            for key_iter in range(10):
                k = 0
                for train_idx, test_idx in validator.split(X, y):
                    print('CLASSIFIER:', "DNN", "DATASET", dataset_names[ds_cat][i],
                          'ITERATION:', key_iter, 'CV_FOLD:', k)
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    df_train = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
                    df_test = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)
                    df_train, df_valid = train_test_split(df_train, test_size=DATA_SPLIT_PCT, random_state=SEED,
                                                          shuffle=True)

                    df_train_1 = df_train[df_train[:, -1] == 1]
                    df_train_2 = df_train[df_train[:, -1] == 2]
                    df_train_1_x = np.delete(df_train_1, -1, axis=1)
                    # df_train_2_x = np.delete(df_train_2, -1, axis=1)

                    df_valid_1 = df_valid[df_valid[:, -1] == 1]
                    df_valid_2 = df_valid[df_valid[:, -1] == 2]
                    df_valid_1_x = np.delete(df_valid_1, -1, axis=1)
                    # df_valid_2_x = np.delete(df_valid_2, -1, axis=1)

                    df_test_1 = df_test[df_test[:, -1] == 1]
                    df_test_2 = df_test[df_test[:, -1] == 2]
                    df_test_1_x = np.delete(df_test_1, -1, axis=1)
                    df_test_2_x = np.delete(df_test_2, -1, axis=1)

                    b.clear_session()
                    au.fit(df_train_1_x, df_valid_1_x)
                    perf_holder = au.predict(df_test)

                    cross_val_pack = [str(dataset_names[ds_cat][i]), key_iter, k, "DNN",
                                      *perf_holder]

                    k = k + 1

                    temp_result.append(cross_val_pack)
                    dh_obj.write_csv(temp_result, configuration['file_level_WPDP_cross_validation_results_des'])


if __name__ == '__main__':
    main()
