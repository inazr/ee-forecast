import pandas as pd  # iloc[Zeilen, Spalten]
import psycopg2

def project_path():
    project_folder = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    project_data = os.path.join(project_folder, 'data')

    return os.path.abspath(project_data)


dl_path, project_path = project_path()

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
    
    df_mosmix_raw_ff = pd.read_sql(
            "select predictiontime, stationid, ff from stg_dwd.mosmix right join stg_dwd.geolocation_ger using (stationid) where predictiontime >= '2018-01-01';",
            con=dwh_conn, parse_dates=True)
    df_mosmix_raw_ff['stationid'] = 'ff_' + df_mosmix_raw_ff['stationid'].astype(str)
    df_mosmix_raw_ff['predictiontime'] = pd.to_datetime(df_mosmix_raw_ff['predictiontime'])
    df_mosmix_pivot_ff = df_mosmix_raw_ff.pivot(index='predictiontime', columns='stationid', values=['ff'])
    
    df_mosmix_pivot = df_mosmix_pivot_ff  # = pd.concat([df_mosmix_pivot_dd, df_mosmix_pivot_ff], axis=1, join='inner')
    
    df_mosmix_pivot.to_csv('df_mosmix.csv')


# Aggregated Generation per Productiontype
def query_entsoe_agppt():
    df_entsoe_agppt = pd.read_sql(
            "select datetime, mapcode, actualgenerationoutput from stg_entsoe.agppt where datetime::date >= '2019-11-01' and resolutioncode = 'PT15M' and mapcode LIKE 'DE_%' and areatypecode = 'CTA' and productiontype = 'Wind Onshore';",
            con=dwh_conn, parse_dates=True)
    
    df_entsoe_agppt = df_entsoe_agppt.pivot(index='datetime', columns='mapcode', values='actualgenerationoutput')
    
    print(df_entsoe_agppt)
    
    df_entsoe_agppt.to_csv('df_entsoe_agppt.csv')


# DayAhead Generation Forecast for Wind and Solar
def query_entsoe_dagfws():
    df_entsoe_dagfws = pd.read_sql(
            "select datetime, mapcode, aggregatedgenerationforecast from stg_entsoe.dagfws where datetime::date >= '2019-11-01' and resolutioncode = 'PT15M' and mapcode LIKE 'DE_%' and areatypecode = 'CTA' and productiontype = 'Wind Onshore';",
            con=dwh_conn, parse_dates=True)
    
    df_entsoe_dagfws = df_entsoe_dagfws.pivot(index='datetime', columns='mapcode',
                                              values='aggregatedgenerationforecast')
    
    print(df_entsoe_dagfws)
    
    df_entsoe_dagfws.to_csv('df_entsoe_dagfws.csv')


if __name__ == "__main__":
    query_mosmix_db()
    query_entsoe_agppt()
    query_entsoe_dagfws()