from keras import models
from keras import layers
from keras.layers.advanced_activations import LeakyReLU
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import pandas as pd  # iloc[Zeilen, Spalten]
import os

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
  
    df_trainingsdata = df_trainingsdata.drop('DE_50HzT', axis=1)
    df_trainingsdata = df_trainingsdata.drop('DE_Amprion', axis=1)
    df_trainingsdata = df_trainingsdata.drop('DE_TenneT_GER', axis=1)
    #df_trainingsdata = df_trainingsdata.drop('DE_TransnetBW', axis=1)
    
    # currently not part of the trainingsdata
    #df_trainingsdata = df_trainingsdata.drop('pred_DE_50HzT', axis=1)
    #df_trainingsdata = df_trainingsdata.drop('pred_DE_Amprion', axis=1)
    #df_trainingsdata = df_trainingsdata.drop('pred_DE_TenneT_GER', axis=1)
    #df_trainingsdata = df_trainingsdata.drop('pred_DE_TransnetBW', axis=1)
    
    return df_trainingsdata


def train_test_split(df_trainingsdata):
    df_trainingsdata = df_trainingsdata.sample(frac=1)

    df_trainingsdata = df_trainingsdata.reindex(sorted(df_trainingsdata.columns), axis=1)
    
    split = 0.999
    split = int(split * len(df_trainingsdata))
    
    train_targets = df_trainingsdata.iloc[:split, 0]
    test_targets = df_trainingsdata.iloc[split:, 0]
    
    
    train_data = df_trainingsdata.iloc[:split, 1:]
    test_data = df_trainingsdata.iloc[split:, 1:]
    
    mean = train_data.mean()
    mean.to_csv(project_data + '\\mean_TBW.csv')
    std = train_data.std()
    std.to_csv(project_data + '\\std_TBW.csv')
    
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    
    train_data = train_data.fillna(0)
    test_data = test_data.fillna(0)
    
    return train_data, train_targets, test_data, test_targets


def build_model_dense(train_data):
    sgd = keras.optimizers.SGD(lr=0.00015, momentum=0.95, nesterov=False, clipnorm=0.99, decay=1e-6)
    model = models.Sequential()
    neurons = 1500
    
    
    # Input Layer
    
    model.add(layers.Dense(neurons, kernel_initializer='he_uniform', input_shape=(train_data.shape[1],)))
    
    for i in range(50):
        #keras.layers.Dropout(0.25, noise_shape=(neurons,), seed=42)
        model.add(LeakyReLU(alpha=0.2))
        model.add(layers.Dense(neurons, kernel_initializer='he_uniform', input_shape=(neurons,)))


    # Output Layer
    model.add(layers.Dense(1, kernel_initializer='he_uniform', activation='linear'))
    
    '''
    Testen:
    loss=get_huber_loss_fn(delta=0.1)
    loss='hinge' <- does not work: investigate!
    tf.losses.Huber(delta=0.9)
    loss=tf.losses.Huber(delta=1.5)
    15232/15232 [==============================] - 9s 587us/step - loss: 69.1568 - mae: 46.8470 - mse: 5909.2832 - val_loss: 154.5919 - val_mae: 103.8075 - val_mse: 25396.6445
    loss=tf.losses.Huber(delta=0.9)
    15232/15232 [==============================] - 9s 570us/step - loss: 34.4352 - mae: 38.7073 - mse: 4616.7876 - val_loss: 87.2887 - val_mae: 97.4355 - val_mse: 19848.4336
    
    custom loss function does not work ;-/ to load model
    
    '''
    model.compile(optimizer=sgd, loss='mae', metrics=['mse'])
    
    return model


def run_keras(train_data, train_targets, test_data, test_targets):
    num_epochs = 400
    model = build_model_dense(train_data)
    
    
    history = model.fit(train_data, train_targets, epochs=num_epochs, batch_size=256, verbose=1,
                        validation_data=(test_data, test_targets))
    
    model.save(project_data + '\\keras_TBW.h5')
    
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