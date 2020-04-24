import psycopg2
import pysftp
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import gc
from importlib.machinery import SourceFileLoader

cnopts = pysftp.CnOpts()
cnopts.hostkeys = None


def project_path():
    project_folder = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    project_data = os.path.join(project_folder, 'data', 'entsoe', 'agppt')

    return os.path.abspath(project_data), os.path.abspath(project_folder)


dl_path, project_path = project_path()

settings = SourceFileLoader("settings", project_path + "/settings.py").load_module()
dwh_conn = settings.dwh_conn
entsoe_ftp = settings.entsoe_ftp

def generate_file_name(current_month, current_year):
    filename = (str(current_year) + '_' + str(current_month) + '_DayAheadGenerationForecastWindSolar.csv')

    return filename


def download_file(filename, dl_path):
    with pysftp.Connection(entsoe_ftp['host'], username=entsoe_ftp['user'], password=entsoe_ftp['password'],
                           cnopts=cnopts) as sftp:
        with sftp.cd(entsoe_ftp['folder'] + 'DayAheadGenerationForecastWindSolar/'):
            sftp.get(filename, dl_path + '/' + filename)


def open_encode_file(filename):
    df_data = pd.read_csv(dl_path + '/' + filename, sep='\t', encoding='UTF-16', skipinitialspace=True)
    df_data.to_csv(dl_path + '/' + filename, index=False, header=False, encoding='utf-8')


def clean_database(current_month, current_year):
    with dwh_conn.cursor() as cur:
        cur.execute("DELETE FROM stg_entsoe.dagfws WHERE year = " + str(current_year) + " AND month = " + str(current_month) + ";")
        dwh_conn.commit()


def load_data_to_db(filename):
    with dwh_conn.cursor() as cur:
        cur.execute("COPY stg_entsoe.dagfws FROM '" + dl_path + '/' + filename + "' WITH (DELIMITER E',', FORMAT CSV);")
        dwh_conn.commit()


def month_year():
    if datetime.now().day < 3:
        if datetime.now().month == 1:
            current_month = datetime.now().month + 11
            current_year = datetime.now().year - 1
        else:
            current_month = datetime.now().month -1
            current_year = datetime.now().year
    else:
            current_month = datetime.now().month
            current_year = datetime.now().year

    return current_month, current_year


if __name__ == "__main__":
    current_month, current_year = month_year()
    filename = generate_file_name(current_month, current_year)
    download_file(filename, dl_path)
    open_encode_file(filename)
    clean_database(current_month, current_year)
    load_data_to_db(filename)
    gc.collect()