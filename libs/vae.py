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
from sklearn.model_selection import train_test_split
from numpy.random import seed
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
from genetic_selection import GeneticSelectionCV
from sklearn.preprocessing import StandardScaler

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)


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


def binerizeCPDP(data):
    for r in range(len(data)):
        if data.iloc[r, -1] > 0:
            data.iloc[r, -1] = 2
        else:
            data.iloc[r, -1] = 1
    return data


SEED = 123
DATA_SPLIT_PCT = 0.25

JM1 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\raw_original_datasets\\KC1\\KC1.csv"

traindata = pd.read_csv(filepath_or_buffer=JM1, index_col=None)
# [0, 1, 3, 10, 13, 31, 6, 18, 19, 17, 20, 21, 22, 32, 33, -1] CM1
# [0, 1, 2, 3, 6, 7, 8, 9, 10, 15, 16, 17, 18, 20, -1] JM1
traindata = traindata.iloc[:, [0, 1, 2, 3, 6, 7, 8, 9, 10, 15, 16, 17, 18, 20, -1]]
# traindata = traindata.drop(
#     [traindata.columns[0]], axis='columns')
# traindata = binerizeCPDP(traindata)

df_train, df_test = train_test_split(traindata, test_size=DATA_SPLIT_PCT, random_state=SEED, shuffle=True)
df_train, df_valid = train_test_split(df_train, test_size=DATA_SPLIT_PCT, random_state=SEED, shuffle=True)

df_train_0 = df_train.loc[traindata['bug'] == 0]
df_train_1 = df_train.loc[traindata['bug'] == 1]
df_train_0_x = df_train_0.drop(['bug'], axis=1)
df_train_1_x = df_train_1.drop(['bug'], axis=1)

df_valid_0 = df_valid.loc[traindata['bug'] == 0]
df_valid_1 = df_valid.loc[traindata['bug'] == 1]
df_valid_0_x = df_valid_0.drop(['bug'], axis=1)
df_valid_1_x = df_valid_1.drop(['bug'], axis=1)

df_test_0 = df_test.loc[traindata['bug'] == 0]
df_test_1 = df_test.loc[traindata['bug'] == 1]
df_test_0_x = df_test_0.drop(['bug'], axis=1)
df_test_1_x = df_test_1.drop(['bug'], axis=1)

# scaler = StandardScaler().fit(df_train_1_x)
# df_train_1_x_rescaled = scaler.transform(df_train_1_x)
# df_valid_1_x_rescaled = scaler.transform(df_valid_1_x)
# df_valid_x_rescaled = scaler.transform(df_valid.drop(['bug'], axis=1))
#
# df_test_1_x_rescaled = scaler.transform(df_test_1_x)
# df_test_x_rescaled = scaler.transform(df_test.drop(['bug'], axis=1))

nb_epoch = 300
batch_size = 20
input_dim = df_train_1_x.shape[1]
encoding_dim = 32
hidden_dim = int(encoding_dim / 2)
learning_rate = 1e-3

# input_layer = Input(shape=(input_dim,))
# encoder = Dense(encoding_dim, activation="sigmoid", activity_regularizer=regularizers.l1(learning_rate))(input_layer)
# encoder = Dense(hidden_dim, activation="sigmoid")(encoder)
# decoder = Dense(hidden_dim, activation='sigmoid')(encoder)
# decoder = Dense(input_dim, activation='sigmoid')(decoder)
# autoencoder = Model(inputs=input_layer, outputs=decoder)

input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='sigmoid')(input_layer)
encoded = Dense(64, activation='sigmoid')(encoded)
encoded = Dense(32, activation='sigmoid')(encoded)

decoded = Dense(32, activation='sigmoid')(encoded)
decoded = Dense(64, activation='sigmoid')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)
autoencoder = Model(inputs=input_layer, outputs=decoded)

autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer='adam')

cp = ModelCheckpoint(filepath="autoencoder_classifier.h5",
                     save_best_only=True,
                     verbose=0)

tb = TensorBoard(log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=True)

history = autoencoder.fit(df_train_0_x, df_train_0_x,
                          epochs=nb_epoch,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(df_valid_0_x, df_valid_0_x),
                          verbose=1,
                          callbacks=[cp, tb]).history

valid_x_predictions = autoencoder.predict(df_valid.iloc[:, 0:-1])
mse = np.mean(np.power(df_valid.iloc[:, 0:-1] - valid_x_predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse,
                         'True_class': df_valid.iloc[:, -1]})

# precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error,
#                                                                pos_label=1)
# plt.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=5)
# plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=5)
# plt.title('Precision and recall for different threshold values on validation data')
# plt.xlabel('Threshold')
# plt.ylabel('Precision/Recall')
# plt.legend()
# plt.show()

test_x_predictions = autoencoder.predict(df_test.iloc[:, 0:-1])
mse = np.mean(np.power(df_test.iloc[:, 0:-1] - test_x_predictions, 2), axis=1)
error_df_test = pd.DataFrame({'Reconstruction_error': mse,
                              'True_class': df_test.iloc[:, -1]})
error_df_test = error_df_test.reset_index()

precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df_test.True_class,
                                                               error_df_test.Reconstruction_error, pos_label=1)
# plt.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=5)
# plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=5)
# plt.title('Precision and recall for different threshold values on test data')
# plt.xlabel('Threshold')
# plt.ylabel('Precision/Recall')
# plt.legend()
# #plt.show()

auc_list = []
accuracy = []
precision = []
recall = []
f1score = []

store = np.zeros((6, len(threshold_rt)))

for i in range(len(threshold_rt)):
    threshold_fixed = threshold_rt[i]
    groups = error_df_test.groupby('True_class')

    # fig, ax = plt.subplots()

    # for name, group in groups:
    #     ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
    #             label="Non-defective" if name == 1 else "Defective")
    # ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
    # ax.legend()
    # plt.title("Reconstruction error for different classes")
    # plt.ylabel("Reconstruction error")
    # plt.xlabel("Data point index")
    # #plt.show()

    pred_y = [1 if e > threshold_fixed else 0 for e in error_df_test.Reconstruction_error.values]

    report = classification_report(error_df_test.True_class, pred_y, output_dict=True)

    # store[0, i] = threshold_fixed
    store[0, i] = round(report['weighted avg']['precision'], 2)
    store[1, i] = round(report['weighted avg']['recall'], 2)
    store[2, i] = round(report['weighted avg']['f1-score'], 2)
    store[3, i] = round(accuracy_score(error_df_test.True_class, pred_y), 2)
    store[4, i] = round(matthews_corrcoef(error_df_test.True_class, pred_y), 2)
    store[5, i] = round(roc_auc_score(error_df_test.True_class, pred_y, average=None), 2)
    # auc_list.append(round(roc_auc_score(error_df_test.True_class, pred_y, average=None), 2))
    # accuracy.append(round(accuracy_score(error_df_test.True_class, pred_y), 2))
    # precision.append(round(report['weighted avg']['precision'], 2))
    # recall.append(round(report['weighted avg']['recall'], 2))
    # f1score.append(round(report['weighted avg']['f1-score'], 2))
a = []
idx = []
for i in range(store.shape[1]):
    a.append(sum(store[:, i]))
    idx.append(i)

[a, idx] = selection_sort(a, idx)
prec = store[:, idx[-1]][0]
reca = store[:, idx[-1]][1]
f1 = store[:, idx[-1]][2]
acc = store[:, idx[-1]][3]
mcc = store[:, idx[-1]][4]
auc = store[:, idx[-1]][5]


