from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Dense, regularizers
from keras.models import Model
from keras.datasets import mnist
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import precision_recall_curve, classification_report
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


def binerizeCPDP(data):
    for r in range(len(data)):
        if data.iloc[r, -1] > 0:
            data.iloc[r, -1] = 2
        else:
            data.iloc[r, -1] = 1
    return data


SEED = 123
DATA_SPLIT_PCT = 0.25

JM1 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\raw_original_datasets\\JM1\\JM1.csv"

traindata = pd.read_csv(filepath_or_buffer=JM1, index_col=None)
# [0, 1, 3, 10, 13, 31, 6, 18, 19, 17, 20, 21, 22, 32, 33, -1] CM1
# [0, 1, 2, 3, 6, 7, 8, 9, 10, 15, 16, 17, 18, 20, -1] JM1
traindata = traindata.iloc[:, [0, 1, 2, 3, 6, 7, 8, 9, 10, 15, 16, 17, 18, 20, -1]]
# traindata = traindata.drop(
#     [traindata.columns[0]], axis='columns')
# traindata = binerizeCPDP(traindata)

df_train, df_test = train_test_split(traindata, test_size=DATA_SPLIT_PCT, random_state=SEED, shuffle=True)
df_train_output = df_train
df_test_output = df_test
## adding noise
df_train = df_train_output + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=df_train_output.shape)
df_test = df_test_output + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=df_test_output.shape)


nb_epoch = 100
batch_size = 12
input_dim = df_train.shape[1]
encoding_dim = 32
hidden_dim = int(encoding_dim / 2)
learning_rate = 1e-3

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(learning_rate))(input_layer)
encoder = Dense(hidden_dim, activation="relu")(encoder)
decoder = Dense(hidden_dim, activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

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

history = autoencoder.fit(df_train, df_train_output,
                          epochs=nb_epoch,
                          batch_size=batch_size,
                          shuffle=True,
                          verbose=1,
                          callbacks=[cp, tb]).history

test_x_predictions = autoencoder.predict(df_test)
mse = np.mean(np.power(df_test - test_x_predictions, 2), axis=1)
error_df_test = pd.DataFrame({'Reconstruction_error': mse,
                              'True_class': df_test['bug']})
error_df_test = error_df_test.reset_index()

threshold_fixed = 0.4
groups = error_df_test.groupby('True_class')

fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label="Non-defective" if name == 1 else "Defective")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

pred_y = [2 if e > threshold_fixed else 1 for e in error_df_test.Reconstruction_error.values]

report = classification_report(error_df_test.True_class, pred_y)
tn, fp, fn, tp = confusion_matrix(error_df_test.True_class, pred_y).ravel()

print(report)
print("TP:", tp, "\n", "FP:", fp, "\n", "FN:", fn, "\n", "TN:", tn)
