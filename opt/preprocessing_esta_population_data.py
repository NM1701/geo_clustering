### e-statデータ前処理（5歳階級別小地域人口データ）

# import
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import glob
import os
pd.set_option('display.max_columns', 200)


# 前処理
def estat_population_data_preprocessing(data_path, csv_path):
    """
    各都道府県のe-Stat人口データを順番に読み込み、結合などの前処理を行う
    """

    # データ読み込み
    estat_file_names = glob.glob(data_path + csv_path)
    estat_list = []
    for file_name in estat_file_names:
        estat = pd.read_csv(file_name, encoding = 'cp932')
        estat_list.append(estat)

    # 都道府県ごとのデータフレームを結合
    estat_df_pre = pd.concat(estat_list)
    estat_df_pre.reset_index(drop = True, inplace = True)

    # 細かく分かれている住所名を結合し、「住所」列に追加
    estat_df_pre['住所'] = estat_df_pre['都道府県名'] + estat_df_pre['市区町村名'] + estat_df_pre['大字・町名'] + estat_df_pre['字・丁目名']
    # 「字・丁目名」がNullの場合、住所もNullになってしまうため、以下コードで追加
    estat_df_pre.fillna({'住所': estat_df_pre['都道府県名'] + estat_df_pre['市区町村名'] + estat_df_pre['大字・町名']}, inplace = True)
    # 「大字・町名」もNullの場合、住所もNullになってしまうため、以下コードで追加
    estat_df_pre.fillna({'住所': estat_df_pre['都道府県名'] + estat_df_pre['市区町村名']}, inplace = True)

    # 必要な行だけを残す
    drop_cols_estat = ['地域階層レベル', '秘匿処理', '秘匿先情報', '合算地域', '総年齢', '平均年齢']
    estat_df = estat_df_pre.drop(drop_cols_estat, axis = 1)

    # 列の並び替え
    col_addr = estat_df.pop('住所')
    estat_df.insert(loc = 7 , column= '住所', value = col_addr)

    # 列名の置換
    def replace_column_name(column_name):
        new_column_name = column_name.replace('歳', '').replace('（再掲）', '').replace('以上', '～')
        return new_column_name

    new_columns = [replace_column_name(col) for col in estat_df.columns]
    estat_df = estat_df.rename(columns = dict(zip(estat_df.columns, new_columns)))

    # NaNの削除
    start_col_name = '総数'
    end_col_name = '20～69'
    def dropna_in_df(df, start_col_name, end_col_name):
        dropna_cols = df.loc[:, start_col_name:end_col_name].columns
        df_dropna = df.dropna(subset = dropna_cols)
        return df

    estat_df = dropna_in_df(estat_df, start_col_name, end_col_name)

    return estat_df


# e-Statデータのフォルダパス
data_path = '../raw_data/男女，年齢（5歳階級）別人口，平均年齢及び総年齢－町丁・字等_2020/'
# csvファイルパス
csv_path = '/*.csv'
# 出力先
output_data_path = '../output/data_preprocessed/e-stat/'
# 前処理
estat_df = estat_population_data_preprocessing(data_path, csv_path)
print(f"estat_df: {estat_df}")
print(f"estat_df.info(): {estat_df.info()}")

# csv出力
os.makedirs(output_data_path, exist_ok = True)
estat_df.to_csv(output_data_path + 'estat_population_2020_preprocessed.csv', index = False, encoding = 'utf-8_sig')

### csv出力後、excel操作で人口比率を算出
