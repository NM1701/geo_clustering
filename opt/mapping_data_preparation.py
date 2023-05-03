## マッピング用データの準備
# - e-Statのシェープファイルをダウンロード（都道府県ごと）
# - 可視化する地図データを1つに統合

# インポート
from urllib.request import urlretrieve
import zipfile
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 200)

def creating_mapping_df(data_info):
    'シェープファイルをダウンロードし、データフレームとして読み込む関数'
    # 展開先フォルダ名
    input_path = data_info[0]
    os.makedirs(input_path, exist_ok = True)
    # R2年 e-stat 国勢調査 小地域（町丁・字等別）
    shapefile_url = data_info[1]
    shapefile_name = data_info[2]
    # ダウンロード
    urlretrieve(url = shapefile_url, filename = shapefile_name)
    # 解凍
    with zipfile.ZipFile(shapefile_name) as existing_zip:
        existing_zip.extractall(input_path)
    # zipファイルの削除
    os.remove(shapefile_name)
    # ファイル名を取得
    files = os.listdir(input_path)
    shapefile = [file for file in files if ".shp" in file][0]

    # 読み込み
    shapefile_path = os.path.join(input_path, shapefile)
    df = gpd.read_file(shapefile_path, encoding = 'cp932')
    # 陸地だけにする
    df = df.loc[df["HCODE"] == 8101, :]

    return df

def data_acquisition_and_merging(data_info_list):
    '複数の緯度経度データのダウンロード・読み込みを行い、1つのデータフレームにまとめる関数'
    # 結合前の各都県の緯度経度データフレーム格納先
    df_list = []
    for data_info in data_info_list:
        # データフレーム生成用のリストに追加
        df = creating_mapping_df(data_info)
        df_list.append(df)
    # 結合
    df = pd.concat(df_list, axis = 0, sort = False, ignore_index = True)

    return df

def preprocessing_mapping_df(df):
    '読み込んだマッピング用データフレームに対する前処理関数'
    # CITY_NAMEやS_NAMEがnullの場所は小地域でない住所（市町村や都道府県しかデータが入っていない場所）のため削除
    df = df.dropna(axis = 0, subset = ['CITY_NAME', 'S_NAME'])
    # クラスタリング結果とのテーブル結合に使う列（住所）を用意
    df['住所'] = df['PREF_NAME'] + df['CITY_NAME'] + df['S_NAME']

    return df

# シェープファイル保存先・DL元のURL・保存ファイル名のリスト
dl_shapefile_path = '../output/data_preprocessed/e-stat/shapefile/'
mapping_data_2020 = [
    [dl_shapefile_path + "shapefile_shiga_2020", "https://www.e-stat.go.jp/gis/statmap-search/data?dlserveyId=A002005212020&code=25&coordSys=1&format=shape&downloadType=5&datum=2011", "shiga_map_2020.zip"],
    [dl_shapefile_path + "shapefile_kyoto_2020", "https://www.e-stat.go.jp/gis/statmap-search/data?dlserveyId=A002005212020&code=26&coordSys=1&format=shape&downloadType=5&datum=2011", "kyoto_map_2020.zip"],
    [dl_shapefile_path + "shapefile_osaka_2020", "https://www.e-stat.go.jp/gis/statmap-search/data?dlserveyId=A002005212020&code=27&coordSys=1&format=shape&downloadType=5&datum=2011", "osaka_map_2020.zip"],
    [dl_shapefile_path + "shapefile_hyogo_2020", "https://www.e-stat.go.jp/gis/statmap-search/data?dlserveyId=A002005212020&code=28&coordSys=1&format=shape&downloadType=5&datum=2011", "hyogo_map_2020.zip"],
    [dl_shapefile_path + "shapefile_nara_2020", "https://www.e-stat.go.jp/gis/statmap-search/data?dlserveyId=A002005212020&code=29&coordSys=1&format=shape&downloadType=5&datum=2011", "nara_map_2020.zip"],
    [dl_shapefile_path + "shapefile_wakayama_2020", "https://www.e-stat.go.jp/gis/statmap-search/data?dlserveyId=A002005212020&code=30&coordSys=1&format=shape&downloadType=5&datum=2011", "wakayama_map_2020.zip"]
]

# 緯度経度データ取得・1つのデータフレームにまとめる
df_mapping_pre = data_acquisition_and_merging(mapping_data_2020)
# 前処理
df_mapping = preprocessing_mapping_df(df_mapping_pre)
# シェープファイルとして保存
shapefile_output_path = '../output/data_preprocessed/e-stat/shapefile/'
os.makedirs(shapefile_output_path, exist_ok = True)
df_mapping.to_file(shapefile_output_path + 'df_mapping_2020.shp', index = False, encoding = 'cp932')
