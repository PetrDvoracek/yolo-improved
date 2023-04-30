import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


import os
import glob

DATACSV = "/home/pedro/datasets/PASCAL_VOC/train.csv"
LBLDIR = "/home/pedro/datasets/PASCAL_VOC/labels"
CLUSTERS = 5


def main():
    df = pd.read_csv(DATACSV)
    width_heights = []
    classes = []
    labels = []
    for _, label_name in df["text"].items():
        label_p = os.path.join(LBLDIR, label_name)
        label_array = np.loadtxt(
            fname=label_p, delimiter=" ", ndmin=2, dtype=np.float32
        )
        labels.append(label_array)
    labels = np.concatenate(labels, axis=0)
    width_heights = labels[:, 3:]
    classes = labels[:, 0]
    unique, counts = np.unique(classes, return_counts=True)
    print("classes: ")
    print(",".join([str(1 / x) for x in counts.tolist()]))
    # for item, count in zip(unique, counts):
    # print(f"{item} : {count}")

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
