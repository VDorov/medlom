import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import  seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def distance(p1, p2):
    return (sum((p1 - p2) ** 2)**0.5)

def main():
    data = pd.read_csv("data_dz.csv")
    scaler = StandardScaler()
    data_scal = scaler.fit_transform(data)
    pca = PCA(n_components=2)
    data2d = pca.fit_transform(data_scal)
    inertias = []
    for k in range(1,21):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scal)
        inertias.append(kmeans.inertia_)


    plt.plot(range(1,21), inertias, 'bo-')

    # for k in range(1,21):
    #     kmeans = KMeans(n_clusters=k)
    #     kmeans.fit(data)
    #     labels = kmeans.labels_
    #     inertias.append(kmeans.inertia_)
    #sns.heatmap(data.corr(),annot=True, cmap="coolwarm")
    # corr_matrix = data.corr()
    # sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    # data.hist()
    plt.show()

   # print(data.corr())

main()

#ограничение в sample
