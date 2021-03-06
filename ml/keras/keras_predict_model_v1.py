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
    mean = pd.read_csv(project_data + '/mean_' + UNB + '_mse_vw.csv', header=0, index_col=0, squeeze=True)
    # print(mean)
    std = pd.read_csv(project_data + '/std_' + UNB + '_mse_vw.csv', header=0, index_col=0, squeeze=True)
    # print(std)

    return mean, std


def get_prediction_data(project_data):
    df_prediction_data = pd.read_sql("with helper_table as ( select max(time_of_prediction) as max_time_of_prediction from stg_dwd.mosmix) select forecast_timestamp, time_of_prediction, stationid, case when ff > 3 and ff < 32 then ((pppp /(ttt*(287.058 /(1-exp((17.5043 * td)/(241.2 + td)-(17.5043 * ttt)/(241.2 + ttt))*((6.112 * exp(17.62 *(ttt-273.15)/(243.12 +(ttt-273.15))))/ pppp)*(1-(287.058 / 461.523))))))/ 2)* ff*ff*ff else 0 end as vw from stg_dwd.mosmix right join ods_dwd.geo_coordinates_ger using (stationid) where (date_part('hour', forecast_timestamp - time_of_prediction ) + 24 * date_part('day', forecast_timestamp - time_of_prediction) - 1) = 0 and forecast_timestamp > '2020-05-10' union all select forecast_timestamp, time_of_prediction, stationid, case when ff > 3 and ff < 32 then ((pppp /(ttt*(287.058 /(1-exp((17.5043 * td)/(241.2 + td)-(17.5043 * ttt)/(241.2 + ttt))*((6.112 * exp(17.62 *(ttt-273.15)/(243.12 +(ttt-273.15))))/ pppp)*(1-(287.058 / 461.523))))))/ 2)* ff*ff*ff else 0 end as vw from stg_dwd.mosmix right join helper_table on max_time_of_prediction = time_of_prediction right join ods_dwd.geo_coordinates_ger using (stationid);",
            con=dwh_conn,
            parse_dates=True)
    print(df_prediction_data)
    
    df_prediction_data = pd.pivot_table(df_prediction_data,
                                        index='forecast_timestamp',
                                        columns='stationid',
                                        values='vw',
                                        dropna=False)
    print(df_prediction_data)

    #df_prediction_data = df_prediction_data.reindex(sorted(df_prediction_data.columns), axis=1)
    
    df_prediction_data = df_prediction_data.resample('15Min').first()

    df_prediction_data.columns = df_prediction_data.columns.astype(str) + '_vw'

    print(df_prediction_data)
    
    return df_prediction_data


def do_ml_magic(project_data, df_prediction_data_UNB, UNB, std, mean):
    df_prediction_data_UNB = df_prediction_data_UNB.interpolate(method='linear', axis=0, limit=7, limit_area='inside')

    df_prediction_data_UNB = df_prediction_data_UNB.fillna(0)
    
    df_prediction_data_UNB = (df_prediction_data_UNB - mean) / std

    df_prediction_data_UNB = df_prediction_data_UNB.round(2)
    df_prediction_data_UNB = df_prediction_data_UNB.fillna(df_prediction_data_UNB.min().min())

    model = load_model(project_data + '/keras_' + UNB + '_mse_vw.h5')

    predictions = model.predict(df_prediction_data_UNB, batch_size=1024, verbose=0, use_multiprocessing=True)
    print(predictions)
    predictions = pd.DataFrame(predictions)
    print(predictions)
    column_name = 'DE_' + UNB + '_FC'
    print(predictions)
    predictions.columns = [column_name]
    print(predictions)
    predictions.index = df_prediction_data_UNB.index
    print(predictions)
    return predictions


def write_to_db(df_all_predictions):
    df_all_predictions.to_csv(project_data + '/predictions/predictions_mse.csv', header=False)
    
    with dwh_conn.cursor() as cur:
        cur.execute(
            "COPY ods_dwd.predictions_mse FROM '" + project_data + "/predictions/predictions_mse.csv' DELIMITER ',' NULL AS '-';")
        dwh_conn.commit()


def plot_graph(df_all_predictions):
    sns.set(rc={'figure.figsize': (16, 9)})
    #sns.palplot(sns.cubehelix_palette(8, start=.5, rot=-.75))
    sns.lineplot(data=df_all_predictions.iloc[:,:4])
    plt.savefig(project_data + '/images/output.png', dpi=72)
    plt.show()


if __name__ == "__main__":
    df_all_predictions = pd.DataFrame()
    
    UNB_list = ['TEN',
                'AMP',
                'TBW',
                '50HzT']

    df_prediction_data = get_prediction_data(project_data)
    
    for UNB in UNB_list:
        mean, std = get_statistics(project_data, UNB)
        predictions = do_ml_magic(project_data, df_prediction_data, UNB, std, mean)
        df_all_predictions = pd.concat([predictions, df_all_predictions], axis=1)

    df_all_predictions['time_of_forecast'] = pd.to_datetime(datetime.today(), format='%Y-%m-%d %H:%M:%S')
    
    print(df_all_predictions)
    
    

    #plot_graph(df_all_predictions)

    write_to_db(df_all_predictions)