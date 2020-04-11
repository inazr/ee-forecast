import pandas as pd  # iloc[Zeilen, Spalten]
from pathlib import Path
from importlib.machinery import SourceFileLoader
import psycopg2
import os

def project_path():
    project_folder = Path(os.path.dirname(os.path.abspath(__file__))).parent
    project_data = os.path.join(project_folder, 'data', 'ml')

    return os.path.abspath(project_data), os.path.abspath(project_folder)


project_data, project_path = project_path()

settings = SourceFileLoader("settings", project_path + "/settings.py").load_module()
dwh_conn = settings.dwh_conn

# MOSMIX data set
def query_mosmix_db():
    '''
    Select predictiontime instead of forecasttime to shift data by -1:
    predictiontime: next hour ,
    forecasttime: last hour,
    forecasttime = predictiontime + 1 hour
    Return valid stationids
    '''
    
    df_mosmix_raw_ff = pd.read_sql("select forecast_timestamp, stationid, ff from stg_dwd.mosmix right join ods_dwd.geo_coordinates_ger using (stationid)",
            con=dwh_conn, parse_dates=True)
    df_mosmix_raw_ff['stationid'] = 'ff_' + df_mosmix_raw_ff['stationid'].astype(str)
    df_mosmix_raw_ff['forecast_timestamp'] = pd.to_datetime(df_mosmix_raw_ff['forecast_timestamp'])
    df_mosmix_pivot_ff = df_mosmix_raw_ff.pivot(index='forecast_timestamp', columns='stationid', values=['ff'])
    
    df_mosmix_pivot = df_mosmix_pivot_ff  # = pd.concat([df_mosmix_pivot_dd, df_mosmix_pivot_ff], axis=1, join='inner')
    
    df_mosmix_pivot.index = df_mosmix_pivot.index.shift(periods=-1, freq='H')  # Done to fix Weather prev hour vs. energy next hour
    
    df_mosmix_pivot.to_csv(project_data + '/df_mosmix.csv')
    print("MOSMIX done")

# Aggregated Generation per Productiontype
def query_entsoe_agppt():
    df_entsoe_agppt = pd.read_sql(
            "select datetime, mapcode, actualgenerationoutput from stg_entsoe.agppt where datetime::date >= '2018-01-01' and resolutioncode = 'PT15M' and mapcode LIKE 'DE_%' and areatypecode = 'CTA' and productiontype = 'Wind Onshore';",
            con=dwh_conn, parse_dates=True)
    
    df_entsoe_agppt = df_entsoe_agppt.pivot(index='datetime', columns='mapcode', values='actualgenerationoutput')
    
    df_entsoe_agppt.to_csv(project_data + '/df_entsoe_agppt.csv')
    print("AGPPT done")
    

# DayAhead Generation Forecast for Wind and Solar
def query_entsoe_dagfws():
    df_entsoe_dagfws = pd.read_sql(
            "select datetime, mapcode, aggregatedgenerationforecast from stg_entsoe.dagfws where datetime::date >= '2018-01-01' and resolutioncode = 'PT15M' and mapcode LIKE 'DE_%' and areatypecode = 'CTA' and productiontype = 'Wind Onshore';",
            con=dwh_conn, parse_dates=True)
    
    df_entsoe_dagfws = df_entsoe_dagfws.pivot(index='datetime', columns='mapcode',
                                              values='aggregatedgenerationforecast')
    
    df_entsoe_dagfws.to_csv(project_data + '/df_entsoe_dagfws.csv')
    print("DAGFWS done")


if __name__ == "__main__":
    query_mosmix_db()
    #query_entsoe_agppt()
    #query_entsoe_dagfws()