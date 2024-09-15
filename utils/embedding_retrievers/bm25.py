
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever
import concurrent.futures
from tqdm import tqdm
from typing import List

def _get_relevant_passage(bm25Retriever:BM25Retriever,
                         query_bundle:QueryBundle):
    
    query:str = query_bundle.query_str
    similar_nodes = bm25Retriever.retrieve(query)
    
    return sorted(similar_nodes, key=lambda x: x.score,reverse=False)

def perform_retrieval(query_bundles,BM25):
    print(f"\nBM-25 Retrieval Progress ...")

    num_completed = 0
    num_queries = len(query_bundles)
    retrieved_results:List[NodeWithScore] = []

    def process_query(query, BM25):
        global num_completed
        retrieved_results.append(_get_relevant_passage(BM25, query))
        num_completed += 1

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for query in query_bundles:
            futures.append(executor.submit(process_query, query, BM25))
            
                # Wait for all threads to complete and update progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_queries):
            pass
    
    print(f"BM-25 retrieval complete")
    return retrieved_results