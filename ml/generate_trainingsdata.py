import pandas as pd  # iloc[Zeilen, Spalten]
from pathlib import Path
import os
pd.options.display.max_rows = 16
pd.options.display.max_columns = 16
pd.set_option('display.width', 1750)

def project_path():
    project_folder = Path(os.path.dirname(os.path.abspath(__file__))).parent
    project_data = os.path.join(project_folder, 'data', 'ml')

    return os.path.abspath(project_data)


project_data = project_path()


def generate_training_data():
    df_entsoe_agppt = pd.read_csv(project_data + '/df_entsoe_agppt.csv', parse_dates=True)
    df_entsoe_agppt = df_entsoe_agppt.set_index(['datetime'])
    df_entsoe_agppt.index.name = 'timestamp'
    
    #print(df_entsoe_agppt)

    #df_entsoe_dagfws = pd.read_csv(project_data + '/df_entsoe_dagfws.csv', parse_dates=True)
    #df_entsoe_dagfws = df_entsoe_dagfws.set_index(['datetime'])
    #df_entsoe_dagfws.index.name = 'timestamp'
    #df_entsoe_dagfws.columns = 'pred_' + df_entsoe_dagfws.columns

    df_mosmix = pd.read_csv(project_data + '/df_mosmix.csv', parse_dates=True, header=1)  # read csv
    df_mosmix = df_mosmix.drop(df_mosmix.index[0])  #
    df_mosmix['stationid'] = pd.to_datetime(df_mosmix['stationid'])
    df_mosmix.rename(columns={'stationid': 'forecast_timestamp'}, inplace=True)
    df_mosmix = df_mosmix.set_index(['forecast_timestamp'])
    df_mosmix.index.name = 'timestamp'

    df_trainingsdata = df_entsoe_agppt.join(other=df_mosmix)

    del df_entsoe_agppt
    del df_mosmix

    '''
    method='cubic' need to test other methods as well!
    Pandas:         https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.interpolate.html
    Scipy:          https://docs.scipy.org/doc/scipy/reference/interpolate.html#univariate-interpolation
    Scipy Tuts:     https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
    '''
    
    df_trainingsdata = df_trainingsdata.interpolate(method='linear', axis=0, limit=7, limit_area='inside')

    df_trainingsdata = df_trainingsdata.dropna(thresh=3000, axis=0)
    df_trainingsdata = df_trainingsdata.round(2)

    print(df_trainingsdata)
    
    df_trainingsdata.to_csv(project_data + '/trainingsdata_pow.csv')


if __name__ == "__main__":
    generate_training_data()