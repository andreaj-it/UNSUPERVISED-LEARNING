import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import decomposition
from sklearn import datasets
from sklearn.cluster import KMeans

from scipy.cluster.vq import kmeans, vq

df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv')

X = df_raw.loc[:, ["MedInc", "Latitude", "Longitude"]]

kmeans = KMeans(n_clusters=6)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")

sns.relplot(
    x="Longitude", y="Latitude", hue="Cluster", data=X, height=6,
);
