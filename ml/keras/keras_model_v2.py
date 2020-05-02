from keras import models
from keras import layers
from keras.layers.core import Activation
import keras

from pathlib import Path
import pandas as pd  # iloc[Zeilen, Spalten]
import os
import numpy as np

pd.options.display.max_rows = 16
pd.options.display.max_columns = 16
pd.set_option('display.width', 2000)


def project_path():
    project_folder = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    project_data = os.path.join(project_folder, 'data', 'ml')
    
    return os.path.abspath(project_data)


project_data = project_path()

# Globals

UNB = ['_TEN',
       '_AMP',
       '_TBW',
       '_50HzT']


def get_trainingsdata():
    df_trainingsdata = pd.read_csv(project_data + '\\trainingsdata.csv', parse_dates=True, index_col=0,
                                   header=0)  # read csv
    df_trainingsdata = df_trainingsdata.iloc[:-1, :]
    
    df_trainingsdata = df_trainingsdata.drop('DE_50HzT', axis=1)
    df_trainingsdata = df_trainingsdata.drop('DE_Amprion', axis=1)
    # df_trainingsdata = df_trainingsdata.drop('DE_TenneT_GER', axis=1)
    df_trainingsdata = df_trainingsdata.drop('DE_TransnetBW', axis=1)
    
    return df_trainingsdata


def train_test_split(df_trainingsdata):
    df_trainingsdata = df_trainingsdata.sample(frac=1)
    
    df_trainingsdata = df_trainingsdata.reindex(sorted(df_trainingsdata.columns), axis=1)
    
    split = 0.8
    split = int(split * len(df_trainingsdata))
    
    train_targets = df_trainingsdata.iloc[:split, 0]
    test_targets = df_trainingsdata.iloc[split:, 0]
    
    train_data = df_trainingsdata.iloc[:split, 1:]
    test_data = df_trainingsdata.iloc[split:, 1:]
    
    mean = train_data.mean()
    mean.to_csv(project_data + '\\mean_TEN_mse_test.csv')
    std = train_data.std()
    std.to_csv(project_data + '\\std_TEN_mse_test.csv')
    
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    
    train_data = train_data.fillna(0)
    test_data = test_data.fillna(0)
    
    return train_data, train_targets, test_data, test_targets


'''
LearningRate & decay:   https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/
                        lr_t = lr_0 * 1 / ( 1 + decay * epoche)

'''


def build_model_dense(train_data):
    sgd = keras.optimizers.SGD(lr=0.000075, momentum=0.95, nesterov=False, clipnorm=2, decay=0.001)
    model = models.Sequential()
    neurons = 2500
    
    # Input Layer
    
    model.add(layers.Dense(neurons, kernel_initializer='he_uniform', input_shape=(train_data.shape[1],)))
    
    for i in range(20):
        #keras.layers.Dropout(0.05, noise_shape=(neurons,), seed=42)
        model.add(Activation('elu'))
        model.add(layers.Dense(neurons, kernel_initializer='he_uniform', input_shape=(neurons,)))
    
    # Output Layer
    model.add(layers.Dense(1, kernel_initializer='he_uniform', activation='linear'))
    
    model.compile(optimizer=sgd, loss='mse', metrics=['mae'])
    
    return model


def run_keras(train_data, train_targets, test_data, test_targets):
    num_epochs = 800
    model = build_model_dense(train_data)
    batch_size = 256
    verbose = 1
    validation_freq = 2
    
    history = model.fit(np.asarray(train_data),
                        np.asarray(train_targets),
                        epochs=num_epochs,
                        batch_size=batch_size,
                        verbose=verbose,
                        shuffle=True,
                        validation_data=(np.asarray(test_data),
                                         np.asarray(test_targets)),
                        validation_freq=validation_freq,
                        use_multiprocessing=True)
    
    model.save(project_data + '\\keras_TEN_mse_test.h5')
    


if __name__ == "__main__":
    df_trainingsdata = get_trainingsdata()
    train_data, train_targets, test_data, test_targets = train_test_split(df_trainingsdata)
    run_keras(train_data, train_targets, test_data, test_targets)