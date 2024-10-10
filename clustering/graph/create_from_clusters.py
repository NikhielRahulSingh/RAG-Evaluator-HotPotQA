import networkx as nx
import numpy as np

class ClusterGraph:

    def __init__(self, clusters, similarity_matrix):
        self.clusters = clusters
        self.similarity_matrix = similarity_matrix

    def get_individual_cluster_graphs(self):

        cluster_graphs = {}  # Dictionary to store each cluster's graph
        
        # Create a graph for each cluster and store it in the dictionary
        for cluster_id, cluster in self.clusters.items():
            cluster_graph = nx.Graph()  # Create a new graph for each cluster
            cluster_graph.add_nodes_from(cluster)
            
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    node_i = cluster[i]
                    node_j = cluster[j]

                    # Get the weight from the similarity matrix
                    weight = self.similarity_matrix[node_i, node_j]

                    # Add the edge with the corresponding weight
                    cluster_graph.add_edge(node_i, node_j, weight=weight)
            
            # Store the cluster graph in the dictionary
            cluster_graphs[cluster_id] = cluster_graph

        return cluster_graphs
    
    def get_combined_cluster_graphs(self):
        individual_cluster_graphs = self.get_individual_cluster_graphs()
        composed_graph = nx.Graph()
        
        for cluster_graph in individual_cluster_graphs.values():
            composed_graph = nx.compose(composed_graph, cluster_graph)

        return composed_graph
    
    def get_connected_graph(self):

        cluster_labels = list(self.clusters.keys())
        graph = self.get_combined_cluster_graphs()
        
        # Iterate through each pair of clusters and add an edge between one representative node
        for i in range(len(cluster_labels)):
            for j in range(i + 1, len(cluster_labels)):
                # Get one representative node from each cluster
                cluster_i = self.clusters[cluster_labels[i]]
                cluster_j = self.clusters[cluster_labels[j]]
                
                node_i = cluster_i[0]  # Select the first node in cluster i
                node_j = cluster_j[0]  # Select the first node in cluster j
                
                # Find the similarity between the representative nodes
                similarity = self.similarity_matrix[node_i, node_j]
                
                # Add an edge between the two nodes based on the similarity
                graph.add_edge(node_i, node_j, weight=similarity)

        return graph
    










