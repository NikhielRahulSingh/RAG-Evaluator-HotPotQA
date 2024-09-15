from langchain_community.embeddings import OllamaEmbeddings
from llama_index.embeddings.ollama import OllamaEmbedding
import os
import pickle
import tqdm

class OllamaEmbedder:

    def __init__(self,model_name,mirostat):
        self.model_name = model_name
        self.mirostat = mirostat

    def get_embeddings(self, chunks, base_url="http://localhost:11434"):

        ollama_embedding = OllamaEmbedding(model_name=self.model_name,
                                           base_url=base_url,
                                           ollama_additional_kwargs={"mirostat": self.mirostat})
        
        embeddings = ollama_embedding.get_text_embedding_batch(chunks, show_progress=True)
        print("embedding complete")

        return embeddings
    
    def save_embeddings(self,model_name,docs,queries,save_dir):

        # Create the "rag_evaluator" folder if it doesn't exist
        save_dir = f"{save_dir}/embeddings/{model_name}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the embedded chunks to a file in the "rag_evaluator" folder
        with open(f"{save_dir}/embedded_chunks.pkl", "wb") as f:
            pickle.dump(docs, f)
            print(f"embedded chunks saved to : {save_dir} as embedded_chunks.pkl")

        # Save the embedded queries to a file in the "rag_evaluator" folder
        with open(f"{save_dir}/embedded_queries.pkl", "wb") as f:
            pickle.dump(queries, f)
            print(f"documents queries saved to : {save_dir} as embedded_queries.pkl")