import pandas as pd
import os
from keras.models import load_model
from datetime import timedelta
from datetime import datetime
import psycopg2
from pathlib import Path
from importlib.machinery import SourceFileLoader

import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

pd.options.display.max_rows = 16
pd.options.display.max_columns = 16
pd.set_option('display.width', 2000)


def project_path():
    project_folder = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    project_data = os.path.join(project_folder, 'data', 'ml')

    return os.path.abspath(project_data), os.path.abspath(project_folder)


project_data, project_path = project_path()

settings = SourceFileLoader("settings", project_path + "/settings.py").load_module()
dwh_conn = settings.dwh_conn


def get_statistics(project_data, UNB):
    mean = pd.read_csv(project_data + '/mean' + UNB + '.csv', header=0, index_col=0, squeeze=True)
    # print(mean)
    std = pd.read_csv(project_data + '/std' + UNB + '.csv', header=0, index_col=0, squeeze=True)
    # print(std)

    return mean, std


def get_prediction_data(project_data):
    df_prediction_data = pd.read_sql("select forecasttime, stationid, ff from stg_dwd.forecastdata right join ods_dwd.geo_coordinates_ger using (stationid) where (forecasttime - time_of_prediction) = '12:00:00';",
            con=dwh_conn, parse_dates=True)
    
    df_prediction_data = pd.pivot_table(df_prediction_data, index='forecasttime', columns='stationid', values='ff')

    df_prediction_data = df_prediction_data.reindex(sorted(df_prediction_data.columns), axis=1)

    df_prediction_data = df_prediction_data.resample('15Min').first()

    df_prediction_data.columns = 'ff_' + df_prediction_data.columns.astype(str)

    return df_prediction_data


def do_ml_magic(project_data, df_prediction_data_UNB, UNB, std, mean):
    df_prediction_data_UNB = (df_prediction_data_UNB - mean) / std

    df_prediction_data_UNB = df_prediction_data_UNB.interpolate(method='linear', axis=0, limit=7, limit_area='inside')

    df_prediction_data_UNB = df_prediction_data_UNB.fillna(0)

    df_prediction_data_UNB = df_prediction_data_UNB.round(2)
    
    model = load_model(project_data + '/keras' + UNB + '.h5')

    predictions = model.predict(df_prediction_data_UNB, batch_size=None, verbose=0)
    predictions = pd.DataFrame(predictions)
    
    column_name = 'DE' + UNB + '_FC'
    
    predictions.columns = [column_name]
    
    predictions.index = df_prediction_data_UNB.index
    
    return predictions


def write_to_db(predictions, prediction_data_folder):
    predictions.to_csv(str(prediction_data_folder) + '/predictions.csv', header=None)
    
    with dwh_conn.cursor() as cur:
        cur.execute(
            "COPY stg_predictions.predictions_keras_model_v1 FROM '" + prediction_data_folder + "/predictions.csv' DELIMITER ',' NULL AS '-';")
        dwh_conn.commit()


def plot_graph(df_all_predictions):
    sns.set(rc={'figure.figsize': (16, 9)})
    #sns.palplot(sns.cubehelix_palette(8, start=.5, rot=-.75))
    sns.lineplot(data=df_all_predictions.iloc[:,:4])
    plt.savefig(project_data + '/images/output.png', dpi=72)
    plt.show()


if __name__ == "__main__":
    df_all_predictions = pd.DataFrame()
    
    UNB_list = ['_TEN',
                '_AMP',
                '_TBW',
                '_50HzT']

    df_prediction_data = get_prediction_data(project_data)
    
    for UNB in UNB_list:
        mean, std = get_statistics(project_data, UNB)
        predictions = do_ml_magic(project_data, df_prediction_data, UNB, std, mean)
        df_all_predictions = pd.concat([predictions, df_all_predictions], axis=1)

    df_all_predictions['time_of_forecast'] = pd.to_datetime(datetime.today(), format='%Y-%m-%d %H:%M:%S')
    
    print(df_all_predictions)

    plot_graph(df_all_predictions)

    #write_to_db(predictions, prediction_data_folder)