from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core.schema import NodeWithScore
from queue import Queue
from typing import List
from llama_index.core import QueryBundle
from llama_index.core.schema import TextNode
import concurrent.futures
from tqdm import tqdm

def calculate_cosine_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

def _get_relevant_passage(graph, 
                         query:QueryBundle, 
                         similarity_threshold):
    
    
    similar_nodes:List[NodeWithScore] = []
    start_node = list(graph.nodes)[0]  # Get the first node in the graph
    visited = set()
    queue = Queue()
    queue.put(start_node)
    visited.add(start_node)
    
    while not queue.empty():
        current_node = queue.get()
        current_vector:List[float] = graph.nodes[current_node]['node'].embedding
        similarity = calculate_cosine_similarity(query.embedding, current_vector)
        
        if similarity >= similarity_threshold:
            node:TextNode = graph.nodes[current_node]['node']
            node_with_score:NodeWithScore = NodeWithScore(node=node, score=similarity)
            similar_nodes.append(node_with_score)
        
        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                neighbor_vector:List[float] = graph.nodes[neighbor]['node'].embedding
                similarity = calculate_cosine_similarity(query.embedding, neighbor_vector)
                
                if similarity >= similarity_threshold:
                    queue.put(neighbor)
                visited.add(neighbor)
    
    return sorted(similar_nodes, key=lambda x: x.score,reverse=False)

def perform_retrieval(query_bundles,db,GRAPH_THRESHOLD):
    print(f"\nBFS Retrieval Progress ...")

    num_completed = 0
    num_queries = len(query_bundles)
    retrieved_results:List[NodeWithScore] = []

    def process_query(query, db):
        global num_completed
        retrieved_results.append(_get_relevant_passage(db, query,similarity_threshold=GRAPH_THRESHOLD))
        num_completed += 1

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for query in query_bundles:
            futures.append(executor.submit(process_query, query, db))
            
                # Wait for all threads to complete and update progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_queries):
            pass
    
    print(f"BFS retrieval complete")
    return retrieved_results