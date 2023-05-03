# ## x-means法によるクラスタリング

import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
from sklearn import cluster, preprocessing
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import matplotlib.pyplot as plt
import japanize_matplotlib
import geopandas as gpd
pd.set_option('display.max_columns', 200)

# データ読込先・出力先パス
data_path = '../output/data_preprocessed/e-stat/'
file_path = 'estat_population_2020_preprocessed_total.xlsx'
output_path = '../output/clustering_results/xmeans/'
shapefile_path = '../output/data_preprocessed/e-stat/shapefile/'
os.makedirs(output_path, exist_ok = True)

# データの読み込み
df = pd.read_excel(data_path + file_path)
print(f"df: {df}")

df = df.copy()
for col in df.loc[:, '総数':'20～69'].columns:
    df[col] = df[col].astype('int')
print(f"df.info(): {df.info()}")

# 必要データだけに絞り込み
df_pre = df.drop(df.columns[0:7], axis = 1)
df_in_use = df_pre.loc[:, '住所':'人口比率_100～']
# 欠損値は0で置換（分母（＝総数）が0で計算できていないエリア）
df_in_use = df_in_use.fillna(0)
# df_in_use

# クラスタリングに使う列を絞り込み
# data_in_use = pd.concat([df_in_use.loc[:, '総数'], df_in_use.loc[:, '人口比率_0～4':'人口比率_100～']], axis = 1)
data_in_use = df_in_use.loc[:, '人口比率_0～4':'人口比率_100～']
data_in_use.describe().to_csv(output_path + "data_in_use_stats.csv", index=False)
print(f"data_in_use.describe(): {data_in_use.describe()}")

# 相関係数
data_in_use.corr().to_csv(output_path + "correlation_matrix.csv", index=False)
print(f"data_in_use.corr(): {data_in_use.corr()}")

# ヒートマップ
heatmap = data_in_use.corr().style.background_gradient(cmap = "bwr", vmin = -1, vmax = 1)
heatmap.savefig(output_path + "correlation_matrix.png")

# 特徴量の配列
X = np.array(data_in_use)

# 特徴量のスケールを合わせる MinMax Scaler
sc_min = -1
sc_max = 1
scaler = preprocessing.MinMaxScaler((sc_min, sc_max))
X_scaled = scaler.fit_transform(X)

# 標準化 Standard Scaler
# scaler = preprocessing.StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Robust Scaler
# scaler = preprocessing.RobustScaler()
# X_scaled = scaler.fit_transform(X)

# x-means法の関数定義
def xmeans_clustering(X, minimal_clusters = 2):
    xm_c = kmeans_plusplus_initializer(X, minimal_clusters).initialize()
    xm_i = xmeans(data = X, initial_centers = xm_c, kmax = 20, ccore = True)
    xmeans_instance = xm_i.process()
    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()

    return xmeans_instance, clusters, centers

# xmeans
xmeans_instance, clusters, centers = xmeans_clustering(X_scaled)
# 分析結果のデータフレーム作成
df_result = pd.DataFrame()
df_result['住所'] = df_in_use.loc[:, '住所']
# クラスター列追加
df_result['cluster'] = 20
for i, cluster in enumerate(clusters):
    df_result.loc[cluster,'cluster'] = i
# 他の変数を結合
df_result = df_result[['住所', 'cluster']].merge(df_in_use, on = '住所', how = 'left')
print(f"df_result: {df_result}")
print(f"df_result['cluster'].value_counts(): {df_result['cluster'].value_counts()}")
# 結果出力
df_result.to_csv(output_path + 'xmeans_result.csv', index = False, encoding = 'utf-8_sig')

# ### 地図上での可視化
# - e-statの統計地理情報システムから無償入手可能な緯度・経度データ（シェープファイル）を用いて可視化を試みる

# シェープファイル読み込み
df_mapping = gpd.read_file(shapefile_path + 'df_mapping_2020.shp', encoding = 'cp932')
# クラスターを追加
df_mapping = df_mapping.merge(df_result.loc[:, ['住所', 'cluster']], on = '住所', how = 'left')
# クラスターが割り当てられていない地域を補完
df_mapping = df_mapping.fillna({"cluster":20})
# cluster列を文字列に変換（カテゴリとして図示するため）
df_mapping['cluster'] = df_mapping['cluster'].astype(int).astype(str)
# クラスター20は"n.a."に置換
df_mapping['cluster'] = df_mapping['cluster'].str.replace('20', 'n.a.')

# クラスタリング結果で色分け（4区分）
plt.rc('legend', fontsize = 32)
df_mapping.plot(column = 'cluster',
        legend = True,
        figsize = [30, 30],
        cmap = 'Set1')
plt.title("x-meansクラスタリング結果可視化", fontsize = 48)
plt.savefig(output_path + 'xmeans_cluster_mapping.jpg', dpi = 300)
# plt.show()

# ### 各変数のクラスターごとの分布を確認

# 可視化用のデータリスト作成
plt.rcParams["font.size"] = 15
data_list = []
title_list = []
hist_columns = df_result.loc[:, '人口比率_0～4':'人口比率_100～'].columns
for col in hist_columns:
    # 積み上げ棒グラフでヒストグラムを描く
    x1 = df_result[df_result['cluster'] == 0][col]
    x2 = df_result[df_result['cluster'] == 1][col]
    x3 = df_result[df_result['cluster'] == 2][col]
    x4 = df_result[df_result['cluster'] == 3][col]
    x5 = df_result[df_result['cluster'] == 4][col]
    data_list.append([x1, x2, x3, x4, x5])

    title = 'Histgram per cluster: ' + col
    title_list.append(title)

# ここからがグラフ作成処理
# グラフのレイアウト（縦・横）
graph_row = 6
graph_col = 4

# グラフ作成
fig = plt.figure(figsize = (graph_col * 7.5, graph_row * 5))
ax_list = []
for j in range(graph_row):
    for i in range(graph_col):
        if (j * graph_col + i) < len(data_list):
            ax_list.append(fig.add_subplot(graph_row, graph_col, j * graph_col + i + 1))
            ax_list[j * graph_col + i].hist(data_list[j * graph_col + i],
                                            bins = 100,
                                            color = ['mediumseagreen', 'tomato', 'royalblue', 'lawngreen', 'moccasin'],
                                            label = ['cluster0', 'cluster1', 'cluster2', 'cluster3', 'cluster4'],
                                            histtype = 'stepfilled',
                                            alpha = 0.7)
            # 書式設定
            ax_list[j * graph_col + i].set_xlabel('value', fontsize = 16)
            ax_list[j * graph_col + i].set_ylabel('frequency', fontsize = 16)
            ax_list[j * graph_col + i].legend(loc ='best', fontsize = 16)
            ax_list[j * graph_col + i].set_title(title_list[j * graph_col + i], fontsize = 18)

            # 増加率の書式設定
#             if 9 <= (j * graph_col + i) <= 17:
#                 ax_list[j * graph_col + i].set_xlim(0, 10)

# グラフ間の間隔指定
plt.subplots_adjust(wspace = 0.4, hspace = 0.6)
# 作成したグラフの保存
plt.savefig(output_path + 'xmeans_clustering_histgram.jpg', dpi = 300)
# グラフをコンソール上に表示
# plt.show()
