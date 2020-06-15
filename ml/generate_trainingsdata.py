import pandas as pd  # iloc[Zeilen, Spalten]
import numpy as np
from pathlib import Path
from pandarallel import pandarallel
import os

pd.options.display.max_rows = 16
pd.options.display.max_columns = 16
pd.set_option('display.width', 1750)

pandarallel.initialize(nb_workers=10,
                       use_memory_fs=False)


def project_path():
    project_folder = Path(os.path.dirname(os.path.abspath(__file__))).parent
    project_data = os.path.join(project_folder, 'data', 'ml')
    
    return os.path.abspath(project_data)


def get_shortest_angle(start, end):
    # 90 -> 180: ((((90 - 180) % 360) + 540) % 360) - 180 -> - 90
    # 180 -> 90: ((((180 - 90) % 360) + 540) % 360) - 180 ->   90
    shortest_angle = ((((start - end) % 360) + 540) % 360) - 180
    shortest_angle = shortest_angle / -4
    
    first = start + shortest_angle
    second = start + shortest_angle * 2
    third = start + shortest_angle * 3
    
    return first, second, third


def circle_interpolation(df):
    for i in range(0, df.shape[0] - 4, 4):  # Zeilen
        if (pd.isna(df.iloc[i])) or (pd.isna(df.iloc[i + 4])):
            continue
        
        df.iloc[i + 1], df.iloc[i + 2], df.iloc[i + 3] = get_shortest_angle(df.iloc[i], df.iloc[i + 4])
    
    return df


def generate_training_data():
    df_entsoe_agppt = pd.read_csv(project_data + '/df_entsoe_agppt.csv', parse_dates=True)
    df_entsoe_agppt = df_entsoe_agppt.set_index(['datetime'])
    df_entsoe_agppt.index.name = 'timestamp'
    
    df_installed_capacity = pd.read_csv(project_data + '/WindOnShore_Installed_Capacity.csv', parse_dates=True,
                                        delimiter=';')
    df_installed_capacity = df_installed_capacity.set_index(['datum'])
    df_installed_capacity = df_installed_capacity.iloc[-3:]
    
    df_entsoe = pd.concat([df_entsoe_agppt, df_installed_capacity], axis=1)
    df_entsoe = df_entsoe.fillna(method='ffill')
    df_entsoe.iloc[:, 4:] = df_entsoe.iloc[:, 4:] / df_entsoe.iloc[:, 4:].max()
    
    df_mosmix_vw = pd.read_csv(project_data + '/df_mosmix_vw.csv', parse_dates=True, header=1)  # read csv
    df_mosmix_vw = df_mosmix_vw.drop(df_mosmix_vw.index[0])  #
    df_mosmix_vw['stationid'] = pd.to_datetime(df_mosmix_vw['stationid'])
    df_mosmix_vw.rename(columns={'stationid': 'forecast_timestamp'}, inplace=True)
    df_mosmix_vw = df_mosmix_vw.set_index(['forecast_timestamp'])
    df_mosmix_vw.index.name = 'timestamp'
    
    df_trainingsdata = df_entsoe.join(other=df_mosmix_vw)
    df_trainingsdata = df_trainingsdata.interpolate(method='linear', axis=0, limit=4, limit_area='inside')
    
    del df_mosmix_vw
    '''
    df_mosmix_dd = pd.read_csv(project_data + '/df_mosmix_dd.csv', parse_dates=True, header=1)  # read csv
    df_mosmix_dd = df_mosmix_dd.drop(df_mosmix_dd.index[0])  #
    df_mosmix_dd['stationid'] = pd.to_datetime(df_mosmix_dd['stationid'])
    df_mosmix_dd.rename(columns={'stationid': 'forecast_timestamp'}, inplace=True)
    df_mosmix_dd = df_mosmix_dd.set_index(['forecast_timestamp'])
    df_mosmix_dd.index.name = 'timestamp'
    
    df_mosmix = pd.DataFrame(index=df_trainingsdata.index)
    df_mosmix_dd = df_mosmix.join(other=df_mosmix_dd).astype(np.float16)
    
    '''
    # https://github.com/nalepae/pandarallel/issues/74 -> Transpose and axis=1 fixes multiplication of rows bug
    '''
    #df_mosmix_dd = df_mosmix_dd.iloc[: , :10]
    
    df_mosmix_dd_cirpiv = df_mosmix_dd.T.parallel_apply(circle_interpolation,
                                               axis=1,
                                               raw=True,
                                               result_type='broadcast').T

    print(df_mosmix_dd_cirpiv)

    '''
    # -> Generate Sinus & Cosinus. Does not work that well...
    '''
    
    df_mosmix_dd_sin = np.sin(df_mosmix_dd_cirpiv).astype(np.float16)
    # df_mosmix_dd_sin.columns = df_mosmix_dd_sin.columns + '_sin'
    
    df_mosmix_dd_cos = np.cos(df_mosmix_dd_cirpiv).astype(np.float16)
    # df_mosmix_dd_cos.columns = df_mosmix_dd_cos.columns + '_cos'

    df_mosmix_dd_sin.columns = df_trainingsdata.columns[8:]
    df_mosmix_dd_cos.columns = df_trainingsdata.columns[8:]
    #df_trainingsdata.columns = df_trainingsdata.columns

    df_mosmix_dd_sin = df_mosmix_dd_sin.astype(np.float16) * df_trainingsdata.iloc[:, 8:].astype(np.float16)
    df_mosmix_dd_cos = df_mosmix_dd_cos.astype(np.float16) * df_trainingsdata.iloc[:, 8:].astype(np.float16)
    
    df_trainingsdata = pd.concat([df_trainingsdata.iloc[:, :8], df_mosmix_dd_sin, df_mosmix_dd_cos], axis=1)
    
    
    
    
    -> OneHot for each quarter of degress
    1: -45 - 45
    2: 45 - 135
    3. 135 - 225
    4. 225 - 315
    

    df_mosmix_dd_cirpiv = ((df_mosmix_dd_cirpiv + 45) % 360) // 90
    print(df_mosmix_dd_cirpiv)

    df_mosmix_dd_cirpiv = df_mosmix_dd_cirpiv.astype('category')
    
    df_mosmix_dd_cirpiv = pd.get_dummies(df_mosmix_dd_cirpiv,
                                         dummy_na=False)
    print(df_mosmix_dd_cirpiv)
    exit()
    df_trainingsdata = pd.concat([df_trainingsdata, df_mosmix_dd_cirpiv], axis=1)
    '''
    
    print(df_trainingsdata)
    
    df_trainingsdata = df_trainingsdata.dropna(thresh=3000, axis=0)
    #df_trainingsdata = df_trainingsdata.dropna(thresh=3000, axis=1)
    df_trainingsdata = df_trainingsdata.round(2)
    
    # Delete the last day as often there is at least one UNB with no data...
    df_trainingsdata = df_trainingsdata.iloc[:-96, :]
    
    df_trainingsdata.to_csv(project_data + '/trainingsdata.csv')
    
    print(df_trainingsdata)
    print("Done...")


if __name__ == "__main__":
    project_data = project_path()
    
    generate_training_data()
