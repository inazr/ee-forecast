from keras import models
from keras import layers
from keras.layers.advanced_activations import LeakyReLU
import keras
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd  # iloc[Zeilen, Spalten]
import os

pd.options.display.max_rows = 16
pd.options.display.max_columns = 16
pd.set_option('display.width', 2000)


def project_path():
    project_folder = Path(os.path.dirname(os.path.abspath(__file__))).parent
    project_data = os.path.join(project_folder, '..', 'data', 'ml')

    return os.path.abspath(project_data)

project_data = project_path()


def get_trainingsdata():
    df_trainingsdata = pd.read_csv(project_data + '\\trainingsdata.csv', parse_dates=True, index_col=0,
                                   header=0)  # read csv
  

    
    df_trainingsdata = df_trainingsdata.drop('DE_50HzT', axis=1)
    df_trainingsdata = df_trainingsdata.drop('DE_Amprion', axis=1)
    # df_trainingsdata = df_trainingsdata.drop('DE_TenneT_GER', axis=1)
    df_trainingsdata = df_trainingsdata.drop('DE_TransnetBW', axis=1)
    
    # currently not part of the trainingsdata
    #df_trainingsdata = df_trainingsdata.drop('pred_DE_50HzT', axis=1)
    #df_trainingsdata = df_trainingsdata.drop('pred_DE_Amprion', axis=1)
    #df_trainingsdata = df_trainingsdata.drop('pred_DE_TenneT_GER', axis=1)
    #df_trainingsdata = df_trainingsdata.drop('pred_DE_TransnetBW', axis=1)
    
    return df_trainingsdata


def train_test_split(df_trainingsdata):
    df_trainingsdata = df_trainingsdata.sample(frac=1)
    
    split = 0.8
    split = int(split * len(df_trainingsdata))
    
    train_targets = df_trainingsdata.iloc[:split, 0]
    test_targets = df_trainingsdata.iloc[split:, 0]
    
    df_trainingsdata = df_trainingsdata.iloc[:, 1:]
    df_trainingsdata = df_trainingsdata.reindex(sorted(df_trainingsdata.columns), axis=1)
    
    train_data = df_trainingsdata.iloc[:split, 0:]
    test_data = df_trainingsdata.iloc[split:, 0:]
    
    mean = train_data.mean()
    mean.to_csv(project_data + 'mean_TEN.csv')
    std = train_data.std()
    std.to_csv(project_data + 'std_TEN.csv')
    
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    
    train_data = train_data.fillna(0)
    test_data = test_data.fillna(0)
    
    print(np.any(np.isnan(train_data)))
    print(np.any(np.isnan(train_targets)))
    print(np.any(np.isnan(test_data)))
    print(np.any(np.isnan(test_targets)))
    
    return train_data, train_targets, test_data, test_targets


def build_model_dense(train_data):
    sgd = keras.optimizers.SGD(lr=0.00025, momentum=0.95, nesterov=False, clipnorm=0.75, decay=1e-6)
    model = models.Sequential()
    
    # Input Layer
    model.add(LeakyReLU(alpha=0.01))
    model.add(
        layers.Dense(train_data.shape[1], kernel_initializer='glorot_uniform', input_shape=(train_data.shape[1],)))
    
    # Hidden Stuff
    model.add(LeakyReLU(alpha=0.01))
    model.add(layers.Dense(806, kernel_initializer='he_uniform', input_shape=(train_data.shape[1],)))
    
    for i in range(50):
        # keras.layers.Dropout(0.1, noise_shape=(806,), seed=42)
        model.add(LeakyReLU(alpha=0.05))
        model.add(layers.Dense(806, kernel_initializer='he_uniform', input_shape=(806,)))
    
    # Output Layer
    model.add(layers.Dense(1, kernel_initializer='he_uniform', activation='linear'))
    
    model.compile(optimizer=sgd, loss='mse', metrics=['mae'])
    
    return model


def run_keras(train_data, train_targets, test_data, test_targets):
    num_epochs = 400
    model = build_model_dense(train_data)
    
    history = model.fit(train_data, train_targets, epochs=num_epochs, batch_size=128, verbose=1,
                        validation_data=(test_data, test_targets))
    
    model.save('keras_TEN.h5')
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    df_trainingsdata = get_trainingsdata()
    train_data, train_targets, test_data, test_targets = train_test_split(df_trainingsdata)
    run_keras(train_data, train_targets, test_data, test_targets)