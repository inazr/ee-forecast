import os
import zipfile
from datetime import datetime
from datetime import timedelta
from ftplib import FTP
from pathlib import Path
import pandas as pd  # iloc[Zeilen, Spalten]
pd.options.mode.chained_assignment = None  # default='warn'
import psycopg2
from lxml import etree
from lxml.etree import XMLParser
from importlib.machinery import SourceFileLoader


ftp_server = 'opendata.dwd.de'
ftp_folder = '/weather/local_forecasts/mos/MOSMIX_S/all_stations/kml/'


def project_path():
    project_folder = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    project_data = os.path.join(project_folder, 'data')

    return os.path.abspath(project_data), os.path.abspath(project_folder)


dl_path, project_path = project_path()

settings = SourceFileLoader("settings", project_path + "/settings.py").load_module()
dwh_conn = settings.dwh_conn


def get_filename(i):
    now = datetime.today() + timedelta(hours=-i)
    filename_kmz = 'MOSMIX_S_' + now.strftime('%Y') + now.strftime('%m') + now.strftime('%d') + now.strftime(
        '%H') + '_240.kmz'
    current_timestamp = now.strftime('%Y') + "-" + now.strftime('%m') + "-" + now.strftime('%d') + " " + now.strftime(
        '%H')+ ":00:00"

    return filename_kmz, current_timestamp, now


def check_filename_exist(filename_kmz):
    if os.path.isfile('/' + dl_path + '/' + filename_kmz):
        print("Filename already exists: " + str(filename_kmz))
        print("Exiting...")
        exit()
    else:
        download_files(filename_kmz)


def download_files(filename_kmz):
    with FTP(ftp_server) as ftp:
        ftp.login()
        ftp.cwd(ftp_folder)
        ftp.retrbinary('RETR ' + filename_kmz, open('/' + dl_path + '/' + filename_kmz, 'wb').write)


def unzip_file(filename_kmz):
    with zipfile.ZipFile(dl_path + '/' + filename_kmz, 'r') as ZipDummy:
        ZipDummy.extractall(dl_path)


def extract_geo_data(filename_kml):
    root = etree.parse(dl_path + '/' + filename_kml, parser=XMLParser(huge_tree=True))

    s_StationIDs = pd.Series(root.xpath("//kml:Document/kml:Placemark/kml:name/text()", namespaces={
        "dwd": "https://opendata.dwd.de/weather/lib/pointforecast_dwd_extension_V1_0.xsd",
        "kml": "http://www.opengis.net/kml/2.2"}))

    s_GeoLocations = pd.Series(root.xpath("//kml:Document/kml:Placemark/kml:Point/kml:coordinates/text()", namespaces={
        "dwd": "https://opendata.dwd.de/weather/lib/pointforecast_dwd_extension_V1_0.xsd",
        "kml": "http://www.opengis.net/kml/2.2"}))

    df_GeoID = pd.concat([s_StationIDs, s_GeoLocations.str.split(",", expand=True)], axis=1)
    df_GeoID.columns = ['StationID', 'lat', 'long', 'height']

    df_GeoID.to_csv(dl_path + '/' + 'geo_coordinates.csv', index=False, header=False)

    return s_StationIDs


def extract_weather_data(filename_kml, s_StationIDs, now):
    root = etree.parse(dl_path + '/' + filename_kml, parser=XMLParser(huge_tree=True))

    df_ForeCastTime = pd.DataFrame(
        root.xpath(
            "//kml:Document/kml:ExtendedData/dwd:ProductDefinition/dwd:ForecastTimeSteps/dwd:TimeStep/text()",
            namespaces={"dwd": "https://opendata.dwd.de/weather/lib/pointforecast_dwd_extension_V1_0.xsd",
                        "kml": "http://www.opengis.net/kml/2.2"}))

    df_ForeCastTime.rename({0: 'ForeCastTime'}, axis='columns', inplace=True)


    df_ForeCastTime['ForeCastTime'] = pd.to_datetime(df_ForeCastTime['ForeCastTime'].astype(str), format='%Y-%m-%d %H:%M:%S')

    '''
    Extracts a dataframe to predict stuff
    '''
    df_ForeCastTime_Predictor = df_ForeCastTime

    df_ForeCastTime['PredictionDate'] = pd.to_datetime(
        now.strftime('%Y') + '-' + now.strftime('%m') + '-' + now.strftime('%d') + ' ' + now.strftime(
            '%H'), format='%Y-%m-%d %H:%M:%S')

    df_ForeCastTime = df_ForeCastTime.iloc[:1, :]

    df_ForeCastData = root.xpath("//kml:Document/kml:Placemark/kml:ExtendedData/dwd:Forecast/dwd:value/text()",
                                 namespaces={
                                     "dwd": "https://opendata.dwd.de/weather/lib/pointforecast_dwd_extension_V1_0.xsd",
                                     "kml": "http://www.opengis.net/kml/2.2"})

    df_ForeCastData = pd.DataFrame(pd.Series(df_ForeCastData).str.split(expand=True))

    '''
    Extracts a dataframe to predict stuff
    '''
    df_ForeCastData_Predictor = df_ForeCastData

    df_ForeCastData = df_ForeCastData.iloc[:, :1]

    df_piv = pd.DataFrame()

    for i in range(0, len(df_ForeCastData), 40):
        df_piv = df_ForeCastData.iloc[i:i + 40].T
        df_piv.columns = [str(x) for x in range(0, 40)]
        df_ForeCastTime['StationID'] = str(s_StationIDs[i / 40])
        df_piv = pd.concat([df_ForeCastTime, df_piv], axis=1)

        with open(dl_path + '/' + 'temp_mosmix.csv', 'a') as f:
            df_piv.to_csv(f, header=False, index=False)

    df_ForeCastData_Predictor = df_ForeCastData_Predictor[
        df_ForeCastData_Predictor.index % 40 == 7]  # Windspeed is the 7th datacolumn

    df_ForeCastData_Predictor = df_ForeCastData_Predictor.reset_index()
    df_ForeCastData_Predictor = df_ForeCastData_Predictor.iloc[:, 1:]

    df_ForeCastData_Predictor = pd.concat([s_StationIDs, df_ForeCastData_Predictor], axis=1)

    df_ForeCastData_Predictor = df_ForeCastData_Predictor.T
    df_ForeCastData_Predictor.columns = df_ForeCastData_Predictor.iloc[0]
    df_ForeCastData_Predictor = df_ForeCastData_Predictor[1:]
    df_ForeCastData_Predictor.index = df_ForeCastTime_Predictor['ForeCastTime']
    df_ForeCastData_Predictor.index = pd.to_datetime(df_ForeCastData_Predictor.index)

    df_ForeCastData_Predictor.to_csv(dl_path + '/' + filename_kml[:-4] + '.csv')

def load_data_to_db():
    with dwh_conn.cursor() as cur:
        cur.execute("COPY stg_dwd.mosmix FROM '" + dl_path + "/temp_mosmix.csv' DELIMITER ',' NULL AS '-';")
        dwh_conn.commit()

        cur.execute("TRUNCATE TABLE stg_dwd.geo_coordinates;")
        cur.execute("COPY stg_dwd.geo_coordinates FROM '" + dl_path + "/geo_coordinates.csv' DELIMITER ',' NULL AS '-';")
        dwh_conn.commit()


def clean_up(filename_kml):
    os.remove(dl_path + '/' + 'temp_mosmix.csv')
    os.remove(dl_path + '/' + 'geo_coordinates.csv')
    os.remove(dl_path + '/' + filename_kml)
    os.remove(dl_path + '/' + filename_kmz)


def check_if_time_of_prediction_on_server(current_timestamp):
    time = pd.read_sql(
        "select time_of_prediction from stg_dwd.mosmix where time_of_prediction = '" + current_timestamp + "';",
        con=dwh_conn, parse_dates=True)

    if time.empty:
        check = True
    else:
        check = False

    return check


if __name__ == "__main__":

    for i in range(4, 9):
        filename_kmz, current_timestamp, now = get_filename(i)

        check = check_if_time_of_prediction_on_server(current_timestamp)
        if check:
            check_filename_exist(filename_kmz)
            unzip_file(filename_kmz)
            filename_kml = filename_kmz[:-1] + 'l'
            s_StationIDs = extract_geo_data(filename_kml)
            extract_weather_data(filename_kml, s_StationIDs, now)
            load_data_to_db()
            clean_up(filename_kml)

        else:
            print("skipping...")





