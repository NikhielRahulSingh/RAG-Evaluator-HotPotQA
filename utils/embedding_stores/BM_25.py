from llama_index.retrievers.bm25 import BM25Retriever

def create_retriever(chunk_nodes,k):

    bm25_retriever = BM25Retriever.from_defaults(nodes=chunk_nodes, similarity_top_k=k)

    print(f"BM-25 Retriever created")
    return bm25_retriever
