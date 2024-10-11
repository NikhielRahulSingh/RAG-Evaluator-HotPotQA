import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import numpy as np

def Agglomerative(similarity_matrix,contexts,threshold):

    dissimilarity_matrix = 1 - similarity_matrix
    clustering = AgglomerativeClustering(n_clusters=None, 
                                         distance_threshold=threshold, 
                                         metric='precomputed', 
                                         linkage='complete')
    clustering.fit(dissimilarity_matrix)
    clusters = {}
    for node, label in zip(contexts, clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(int(node))

    return clusters