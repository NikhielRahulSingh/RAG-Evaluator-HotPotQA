import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from queue import PriorityQueue
from llama_index.core import QueryBundle
from llama_index.core.schema import TextNode
from typing import List
from llama_index.core.schema import NodeWithScore
import math
import concurrent.futures
from tqdm import tqdm
import torch
from sentence_transformers import util

def euclidean_distance(vec1, vec2):
    return math.sqrt(sum((v1 - v2) ** 2 for v1, v2 in zip(vec1, vec2)))

def calculate_cosine_similarity(vec1, vec2):
    tensor1 = torch.tensor(vec1)
    tensor2 = torch.tensor(vec2)
    sim = util.cos_sim(tensor1, tensor2)
    return sim

G = None
THRESHOLD = None

def retrieve_contexts(question:QueryBundle, 
                      similarity_threshold,
                      heuristic=euclidean_distance):
    
    global G

    similar_nodes:List[NodeWithScore] = []
    start_node = list(G.nodes)[0]
    visited = set()
    priority_queue = PriorityQueue()

    # Initialize the priority queue with the start node
    priority_queue.put((0, start_node))
    visited.add(start_node)
    
    while not priority_queue.empty():
        _, current_node = priority_queue.get()
        node_data = G.nodes[current_node]['node']
        similarity = calculate_cosine_similarity(question.embedding, node_data.embedding)
        
        if similarity >= similarity_threshold:
            similar_nodes.append(NodeWithScore(node=node_data, score=similarity))
        
        for neighbor in G.neighbors(current_node):
            if neighbor not in visited:
                neighbor_node_data = G.nodes[neighbor]['node']
                similarity = calculate_cosine_similarity(question.embedding, neighbor_node_data.embedding)
                
                if similarity >= similarity_threshold:
                    priority_queue.put((heuristic(neighbor_node_data.embedding, question.embedding), neighbor))
                visited.add(neighbor)

    sorted_nodes:List[NodeWithScore] = sorted(similar_nodes, key=lambda x: x.score,reverse=True)
    

    return sorted_nodes

def perform_retrieval(hotpot_qa_df, graph_db,threshold):
    print(f"\nA Star retrieval in progress...")

    global G
    G = graph_db
    
    global THRESHOLD
    THRESHOLD = threshold

    retrieved_contexts = []

    for question in tqdm(hotpot_qa_df['question'], desc="Retrieving contexts", unit="question"):
        result = retrieve_contexts(question,THRESHOLD)
        retrieved_contexts.append(result)
    
    print(f"A* retrieval complete")
    return retrieved_contexts