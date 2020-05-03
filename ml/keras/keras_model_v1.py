from keras import models
from keras import layers
from keras.layers.core import Activation
import keras
from pathlib import Path
import pandas as pd  # iloc[Zeilen, Spalten]
import os
import numpy as np
from keras.callbacks import ModelCheckpoint

pd.options.display.max_rows = 16
pd.options.display.max_columns = 16
pd.set_option('display.width', 2000)


def project_path():
    project_folder = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    project_data = os.path.join(project_folder, 'data', 'ml')

    return os.path.abspath(project_data)


def get_trainingsdata(UNB):
    # Loading a the trainingsdata
    df_trainingsdata = pd.read_csv(project_data + '\\trainingsdata.csv', parse_dates=True, index_col=0,
                                   header=0)  # read csv
  
    '''
    Deleting the unused UNB production data, keeping only 1 column as targets
    '''
    if UNB == '_TEN'
        df_trainingsdata = df_trainingsdata.drop('DE_50HzT', axis=1)
        df_trainingsdata = df_trainingsdata.drop('DE_Amprion', axis=1)
        df_trainingsdata = df_trainingsdata.drop('DE_TransnetBW', axis=1)
    
    if UNB == '_AMP'
        df_trainingsdata = df_trainingsdata.drop('DE_50HzT', axis=1)
        df_trainingsdata = df_trainingsdata.drop('DE_TransnetBW', axis=1)
        df_trainingsdata = df_trainingsdata.drop('DE_TenneT_GER', axis=1)

    if UNB == '_TBW'
        df_trainingsdata = df_trainingsdata.drop('DE_50HzT', axis=1)
        df_trainingsdata = df_trainingsdata.drop('DE_Amprion', axis=1)
        df_trainingsdata = df_trainingsdata.drop('DE_TenneT_GER', axis=1)

    if UNB == '_50HzT'
        df_trainingsdata = df_trainingsdata.drop('DE_Amprion', axis=1)
        df_trainingsdata = df_trainingsdata.drop('DE_TenneT_GER', axis=1)
        df_trainingsdata = df_trainingsdata.drop('DE_TransnetBW', axis=1)
    
    return df_trainingsdata


def train_test_split(df_trainingsdata, UNB):
    # Shuffle the trainingsdata
    df_trainingsdata = df_trainingsdata.sample(frac=1)
    
    # Order all columns
    df_trainingsdata = df_trainingsdata.reindex(sorted(df_trainingsdata.columns), axis=1)
    
    # Train / Test Split
    split = 0.8
    split = int(split * len(df_trainingsdata))
    
    train_targets = df_trainingsdata.iloc[:split, 0]
    test_targets = df_trainingsdata.iloc[split:, 0]
    
    train_data = df_trainingsdata.iloc[:split, 1:]
    test_data = df_trainingsdata.iloc[split:, 1:]
    
    mean = train_data.mean()
    mean.to_csv(project_data + '\\mean' + UNB + '_mse_vw.csv')
    
    std = train_data.std()
    std.to_csv(project_data + '\\std' + UNB + '_mse_vw.csv')

    max = train_data.max().max()
    max.to_csv(project_data + '\\max' + UNB + '_mse_vw.csv')

    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    
    # train_data = train_data.fillna(0)
    # test_data = test_data.fillna(0)
    
    return train_data, train_targets, test_data, test_targets


def build_model_dense(train_data):
    # Optimizer
    sgd = keras.optimizers.SGD(lr=0.000075,
                               momentum=0.95,
                               nesterov=False,
                               clipnorm=2,
                               decay=0.001)
    
    # Number of Neurons
    neurons = 1600

    # Modeltype
    model = models.Sequential()
    
    # Input Layer
    model.add(layers.Dense(neurons,
                           kernel_initializer='he_uniform',
                           input_shape=(train_data.shape[1],)))
    
    # Hidden Stuff
    for i in range(50):
        # keras.layers.Dropout(0.05, noise_shape=(neurons,), seed=42)
        model.add(Activation('elu'))
        model.add(layers.Dense(neurons,
                               kernel_initializer='he_uniform',
                               input_shape=(neurons,)))

    # Output Layer
    model.add(layers.Dense(1,
                           kernel_initializer='he_uniform',
                           activation='linear'))
    
    model.compile(optimizer=sgd,
                  loss='mse',
                  metrics=['mae'])

    return model


def run_keras(train_data, train_targets, test_data, test_targets, UNB):
    model = build_model_dense(train_data)
    
    # Settings
    num_epochs = 800
    batch_size = 256
    verbose = 1
    validation_freq = 10
    
    # Saving the model if val_loss has improved during the last n epochs
    checkpoint = ModelCheckpoint(project_data + '\\keras' + UNB + '_mse_vw.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=50)

    callbacks_list = [checkpoint]
    
    history = model.fit(np.asarray(train_data),
                        np.asarray(train_targets),
                        epochs=num_epochs,
                        batch_size=batch_size,
                        verbose=verbose,
                        shuffle=True,
                        validation_data=(np.asarray(test_data),
                                         np.asarray(test_targets)),
                        validation_freq=validation_freq,
                        use_multiprocessing=True,
                        callbacks = callbacks_list)
    

if __name__ == "__main__":
    project_data = project_path()
    
    UNBS = ['_TEN',
            '_AMP',
            '_TBW',
            '_50HzT']
    
    for UNB in UNBS:
        df_trainingsdata = get_trainingsdata(UNB)
        train_data, train_targets, test_data, test_targets = train_test_split(df_trainingsdata, UNB)
        run_keras(train_data, train_targets, test_data, test_targets, UNB)