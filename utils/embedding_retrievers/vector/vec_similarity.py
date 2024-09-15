from llama_index.core.schema import TextNode,NodeWithScore
from typing import List
import concurrent.futures
from tqdm import tqdm

V = None

def retrieve_contexts(question):
    global V
    response = V.query(query_embeddings=[question.embedding],n_results=10)
    _ = response["ids"][0]
    return _

def perform_retrieval(hotpot_qa_df,vector_db):
    global V
    V = vector_db

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(retrieve_contexts, question) for question in hotpot_qa_df['question']]
        progress_bar = tqdm(total=len(futures))
        concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
        progress_bar.update(len(futures))
        retrieved_contexts = [future.result() for future in futures]

    print(f"Vector Similarity retrieval complete")
    return retrieved_contexts

