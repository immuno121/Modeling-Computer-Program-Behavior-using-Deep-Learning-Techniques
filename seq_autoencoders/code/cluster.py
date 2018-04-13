from __future__ import print_function
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
def cluster():


    X=np.load('../result/latent_dims.npy')
    print (X.shape)
    n_clusters=8
    kmeans = KMeans(n_clusters=8, random_state=0).fit(X)
    print(kmeans.labels_)
    target=kmeans.labels_
    target=np.array(target)

    fmap=open("file_index_map.txt", "r")
    lmap=fmap.readlines()
    lmap=np.array(lmap)

    with open("file_cluster_map.txt", "w") as f:
        for i in range(n_clusters):
            f.write('cluster_'+str(i)+': ')
            indices=np.where(target==i)[0]
            arr=lmap[indices]
            for j in range(len(arr)):
                f.write(arr[j] + "\n")
            #k=k+1


    #import matplotlib.pyplot as plt
    #pca = PCA(n_components=50)
    #pca_result = pca.fit_transform(X)


   

    

    #n_sne = 7000

    #tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    #tsne_results = tsne.fit_transform(pca_result)

    #print (tsne_results.shape)
    #print(tsne_results[0])
    #print(tsne_results[1])
    #t_x=tsne_results[:,0]
    #t_y=tsne_results[:,1]




   # plt.scatter(t_x,t_y,c=target)
   # plt.show()
