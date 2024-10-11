import numpy as np
import statistics
from typing import Dict

class ClusterAnalysis:

    def __init__(self, 
                 clusters:Dict[int,list[int]], 
                 similarity_matrix):
        self.clusters = clusters
        self.similarity_matrix = similarity_matrix
        self.avg_cluster_similarity: Dict[int,float] = {}

    def _get_avg_cluster_variance(self,cluster:list[int]) -> float:
        variances = []
        for i, node in enumerate(cluster):
            similarities = []
            for j, other_node in enumerate(cluster):
                if i != j: 
                    cosine_sim = self.similarity_matrix[i][j]
                    similarities.append(cosine_sim)
            if len(similarities) < 2:
                variances.append(0)
            else:
                variance = statistics.variance(similarities)
                variances.append(variance)

        return statistics.mean(variances)
    
    def _get_avg_cluster_similarity(self,cluster:list[int]):
        avg_similarities = []
        for i, node in enumerate(cluster):
            similarities = []
            for j, other_node in enumerate(cluster):
                if i != j: 
                    cosine_sim = self.similarity_matrix[i][j]
                    similarities.append(cosine_sim)
            if len(similarities) == 0:
                avg_similarities.append(0)
            else:
                avg_similarities.append(statistics.mean(similarities))
        
        return statistics.mean(avg_similarities)

    def get_avg_variance_for_each_cluster(self) -> Dict[int,float]:
        cluster_variance:Dict[int,float] = {}
        for key in self.clusters.keys():
            cluster_variance[key] = self._get_avg_cluster_variance(self.clusters[key])
        
        return cluster_variance
    
    def get_mean_avg_cluster_similarity_for_each_cluster(self) -> Dict[int,float]:
        cluster_similarity:Dict[int,float] = {}
        for key in self.clusters.keys():
            cluster_similarity[key] = self._get_avg_cluster_similarity(self.clusters[key])
        
        return cluster_similarity