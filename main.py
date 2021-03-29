from sklearn.cluster import KMeans
from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mns = [(5,3),(15,3),(10,8)]
    scales = [(2,1), (1,1), (1,2)]
    params = zip(mns, scales)

  #  g,h=zip(*params)
  #  print(g)
#    params = zip(mns, scales)


    clusters = []

    for parset in params:

        dist_x = norm(loc=parset[0][0], scale=parset[1][0])
        dist_y = norm(loc=parset[0][1], scale=parset[1][1])
        cluster_x = dist_x.rvs(size=100)
        cluster_y = dist_y.rvs(size=100)
        cluster= zip(cluster_x,cluster_y)
        clusters.extend(cluster)
    print(type(clusters))
    x,y = zip(*clusters)
    plt.figure()
    plt.scatter(x,y)
    plt.title('Punkty 2D', fontsize=14)
    plt.tight_layout()
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
  #  plt.show()

    print(type(x))
    print(type(clusters))
    cluserer = KMeans(n_clusters=3)

    X=np.array(clusters)
    print(type(X))
    y_pred=cluserer.fit_predict(X)
    print(y_pred)
    print(type(y_pred))
    red=y_pred == 0
    blue= y_pred == 1
    cyan = y_pred == 2

    plt.figure()
    plt.scatter(X[red,0], X[red,1], c="r")
    plt.scatter(X[blue, 0], X[blue, 1], c="b")
    plt.scatter(X[cyan, 0], X[cyan, 1], c="c")
    plt.title('Punkty 2D 2', fontsize=14)
    plt.tight_layout()
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
