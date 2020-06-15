from keras import models
from keras import layers
from keras.layers.core import Activation
import keras
from pathlib import Path
import pandas as pd  # iloc[Zeilen, Spalten]
import os
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping


pd.options.display.max_rows = 16
pd.options.display.max_columns = 16
pd.set_option('display.width', 2000)


def project_path():
    project_folder = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    project_data = os.path.join(project_folder, 'data', 'ml')

    return os.path.abspath(project_data)


def get_trainingsdata(UNB):
    # Loading the trainingsdata
    df_trainingsdata = pd.read_csv(project_data + '\\trainingsdata.csv',
                                   parse_dates=True,
                                   index_col=0,
                                   header=0)
    
    # df_trainingsdata = df_trainingsdata.iloc[:, :3221]
    

    '''
    0:4 -> Windproduction
    4:8 -> Installed Capacity
    8:3221 -> kinetic energy
    3221:6434 -> sin dd
    6234:9647 -> cos dd
    '''

    #df_trainingsdata = df_trainingsdata.iloc[:, :3221]

    #df_sin = df_trainingsdata.iloc[:, 3221:6434].mean(axis=1)
    #df_cos = df_trainingsdata.iloc[:, 6234:9647].mean(axis=1)
    #df_trainingsdata = df_trainingsdata.iloc[: , :3215]
    #df_trainingsdata = pd.concat([df_trainingsdata, df_sin, df_cos], axis=1)
  
    '''
    Deleting the unused UNB production data, keeping only 1 column as targets
    '''
    if UNB == 'TEN':
        df_trainingsdata = df_trainingsdata.drop('DE_50HzT', axis=1)
        df_trainingsdata = df_trainingsdata.drop('DE_Amprion', axis=1)
        df_trainingsdata = df_trainingsdata.drop('DE_TransnetBW', axis=1)

        df_trainingsdata = df_trainingsdata.drop('50HzT', axis=1)
        df_trainingsdata = df_trainingsdata.drop('AMP', axis=1)
        df_trainingsdata = df_trainingsdata.drop('TBW', axis=1)
        df_trainingsdata = df_trainingsdata.drop('TEN', axis=1)
    
        df_trainingsdata = df_trainingsdata.dropna(subset=['DE_TenneT_GER'], axis=0, how='any')
        
    if UNB == 'AMP':
        df_trainingsdata = df_trainingsdata.drop('DE_50HzT', axis=1)
        df_trainingsdata = df_trainingsdata.drop('DE_TransnetBW', axis=1)
        df_trainingsdata = df_trainingsdata.drop('DE_TenneT_GER', axis=1)
        
        df_trainingsdata = df_trainingsdata.drop('50HzT', axis=1)
        df_trainingsdata = df_trainingsdata.drop('AMP', axis=1)
        df_trainingsdata = df_trainingsdata.drop('TBW', axis=1)
        df_trainingsdata = df_trainingsdata.drop('TEN', axis=1)

        df_trainingsdata = df_trainingsdata.dropna(subset=['DE_Amprion'], axis=0, how='any')

    if UNB == 'TBW':
        df_trainingsdata = df_trainingsdata.drop('DE_50HzT', axis=1)
        df_trainingsdata = df_trainingsdata.drop('DE_Amprion', axis=1)
        df_trainingsdata = df_trainingsdata.drop('DE_TenneT_GER', axis=1)
        
        df_trainingsdata = df_trainingsdata.drop('50HzT', axis=1)
        df_trainingsdata = df_trainingsdata.drop('AMP', axis=1)
        df_trainingsdata = df_trainingsdata.drop('TBW', axis=1)
        df_trainingsdata = df_trainingsdata.drop('TEN', axis=1)

        df_trainingsdata = df_trainingsdata.dropna(subset=['DE_TransnetBW'], axis=0, how='any')


    if UNB == '50HzT':
        df_trainingsdata = df_trainingsdata.drop('DE_Amprion', axis=1)
        df_trainingsdata = df_trainingsdata.drop('DE_TenneT_GER', axis=1)
        df_trainingsdata = df_trainingsdata.drop('DE_TransnetBW', axis=1)
        
        df_trainingsdata = df_trainingsdata.drop('50HzT', axis=1)
        df_trainingsdata = df_trainingsdata.drop('AMP', axis=1)
        df_trainingsdata = df_trainingsdata.drop('TBW', axis=1)
        df_trainingsdata = df_trainingsdata.drop('TEN', axis=1)

        df_trainingsdata = df_trainingsdata.dropna(subset=['DE_50HzT'], axis=0, how='any')

    df_trainingsdata = df_trainingsdata.replace([np.inf, -np.inf], np.nan)
    print(df_trainingsdata)
    
    return df_trainingsdata


def train_test_split(df_trainingsdata, UNB):
    # Shuffle the trainingsdata
    df_trainingsdata = df_trainingsdata.sample(frac=1)
    
    # Order all columns
    # df_trainingsdata = df_trainingsdata.reindex(sorted(df_trainingsdata.columns), axis=1)
    
    # Train / Test Split
    split = 0.9
    split = int(split * len(df_trainingsdata))
    
    train_targets = df_trainingsdata.iloc[:split, 0]
    test_targets = df_trainingsdata.iloc[split:, 0]
    
    train_data = df_trainingsdata.iloc[:split, 1:]
    test_data = df_trainingsdata.iloc[split:, 1:]
    
    '''
    Stardardization
    '''
    mean = train_data.mean()
    mean.to_csv(project_data + '\\mean_' + UNB + '_mse_vw.csv')
    
    std = train_data.std()
    std.to_csv(project_data + '\\std_' + UNB + '_mse_vw.csv')

    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    
    train_data = train_data.fillna(0)
    test_data = test_data.fillna(0)

    train_targets = train_targets.fillna(0)
    test_targets = test_targets.fillna(0)
    
    return train_data, train_targets, test_data, test_targets


def build_model(train_data):
    # Optimizer
    sgd = keras.optimizers.SGD(lr=0.1,
                               momentum=0.95,
                               nesterov=False,
                               clipnorm=1
                               # , decay=0.00001
                               )
    
    # Number of Neurons
    neurons = 1600

    # Modeltype
    model = models.Sequential()
    
    # Input Layer
    model.add(layers.Dense(neurons,
                           kernel_initializer='he_uniform',
                           input_shape=(train_data.shape[1],)))
    
    # Hidden Stuff
    for i in range(40):
        keras.layers.Dropout(0.05, noise_shape=(neurons,), seed=42)
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
    '''
    Learning Rate: https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/
    '''
    
    model = build_model(train_data)
    
    # Settings
    num_epochs = 500
    batch_size = 512
    validation_freq = 10
    k = 5
    num_val_samples = len(train_data) // k
    all_scores = []
    
    # Saving the model if val_loss has improved during the last n epochs
    checkpoint = ModelCheckpoint(project_data + '\\keras_' + UNB + '_mse_vw.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=validation_freq)
    
    rlrop = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.25,
                              patience=4,
                              verbose=1)
    
    earlystop = EarlyStopping(monitor="val_loss",
                              min_delta=0,
                              patience=13,
                              verbose=1,
                              mode="auto",
                              baseline=None,
                              restore_best_weights=False)

    callbacks_list = [checkpoint, rlrop, earlystop]
    
    for i in range(k):
        print('processing fold #', i)
        
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

        partial_train_data = np.concatenate(
                [train_data[:i * num_val_samples],
                 train_data[(i + 1) * num_val_samples:]],
                axis=0)
        partial_train_targets = np.concatenate(
                [train_targets[:i * num_val_samples],
                 train_targets[(i + 1) * num_val_samples:]],
                axis=0)
        
        model.fit(np.asarray(partial_train_data),
                  np.asarray(partial_train_targets),
                  epochs=num_epochs,
                  batch_size=batch_size,
                  verbose=0,
                  shuffle=True,
                  validation_data=(np.asarray(val_data),
                                   np.asarray(val_targets)),
                  validation_freq=validation_freq,
                  use_multiprocessing=True,
                  callbacks=callbacks_list)

        val_mse, val_mae = model.evaluate(test_data, test_targets, verbose=0)
        all_scores.append(val_mae)
        all_scores.append(val_mse)
        
        print(all_scores)


    
    
if __name__ == "__main__":
    project_data = project_path()
    
    UNBS = ['50HzT',
            'TEN',
            'TBW',
            'AMP']
    
    UNB = UNBS[0]

    df_trainingsdata = get_trainingsdata(UNB)
    train_data, train_targets, test_data, test_targets = train_test_split(df_trainingsdata, UNB)
    run_keras(train_data, train_targets, test_data, test_targets, UNB)
    

    # os.system("shutdown /s /t 1")