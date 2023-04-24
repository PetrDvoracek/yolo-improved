import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


import os
import glob

DATACSV = "/home/pedro/datasets/PASCAL_VOC/train.csv"
LBLDIR = "/home/pedro/datasets/PASCAL_VOC/labels"
CLUSTERS = 3


def main():
    df = pd.read_csv(DATACSV)
    width_heights = []
    for _, label_name in df["text"].items():
        label_p = os.path.join(LBLDIR, label_name)
        width_heights.append(
            np.loadtxt(fname=label_p, delimiter=" ", ndmin=2, dtype=np.float32)[:, 3:]
        )
    width_heights = np.concatenate(width_heights, axis=0)
    areas = width_heights[:, 0] * width_heights[:, 1]
    kmeans = KMeans(n_clusters=CLUSTERS).fit(areas.reshape(-1, 1))
    print(
        "areas: \n",
        ",".join([str(x) for x in kmeans.cluster_centers_.reshape(-1).tolist()]),
    )

    ratios = width_heights[:, 0] / width_heights[:, 1]
    kmeans = KMeans(n_clusters=CLUSTERS).fit(ratios.reshape(-1, 1))
    print(
        "ratios: \n",
        ",".join([str(x) for x in kmeans.cluster_centers_.reshape(-1).tolist()]),
    )

    plt.scatter(areas, np.zeros(len(areas)))
    plt.savefig("./areas.png")

    plt.scatter(ratios, np.zeros(len(ratios)))
    plt.savefig("./ratios.png")


if __name__ == "__main__":
    main()
