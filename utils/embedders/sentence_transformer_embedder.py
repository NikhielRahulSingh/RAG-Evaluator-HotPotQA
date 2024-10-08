from sentence_transformers import SentenceTransformer
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import login
from llama_index.core.schema import TextNode
from llama_index.core.schema import QueryBundle
from typing import List,Union

class SentenceTransformerEmbedder:

    def __init__(self, huggingface_token, cache_loc, model_save_loc):
        self.huggingface_token = huggingface_token
        self.cache_loc = cache_loc
        self.model_save_loc = model_save_loc

    def download_embedding_model(self,model_name:str):
        
        login(token=self.huggingface_token)
        embedding_model = SentenceTransformer(model_name,
                                              cache_folder=self.cache_loc,
                                              trust_remote_code=True
                                             )
        embedding_model.save(path=f'{self.model_save_loc}/{model_name}', safe_serialization=True)

        print(f'Done downloading {model_name} to {self.model_save_loc}')

    def save_embeddings(self,
                        contexts,
                        df,
                        df_name,
                        similarity_matrix,
                        save_dir):

        # Create the "rag_evaluator" folder if it doesn't exist
        save_dir = f"{save_dir}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the embedded chunks to a file in the "rag_evaluator" folder
        with open(f"{save_dir}/contexts.pkl", "wb") as f:
            pickle.dump(contexts, f)
            print(f"embedded chunks saved to : {save_dir} as contexts.pkl")

        # Save the embedded queries to a file in the "rag_evaluator" folder
        df.to_pickle(f'{save_dir}/{df_name}.pkl')
        print(f"{df_name} saved to : {save_dir} as embedded_queries.pkl")

        # Save the similarity matrix to a file in the "rag_evaluator" folder
        with open(f"{save_dir}/similarity_matrix.pkl", "wb") as f:
            pickle.dump(similarity_matrix, f)
            print(f"similarity matrix saved to : {save_dir} as similarity_matrix.pkl")

    def get_embeddings(self, 
                       model,
                       documents:List[Union[TextNode, QueryBundle]],
                       similarity_matrix:bool):
    
        chunks = [(doc.text if isinstance(doc, TextNode) else doc.query_str) for doc in documents]
        embeddings = model.encode(chunks,show_progress_bar=True)

        def assign_embedding(doc, embedding):
             doc.embedding = embedding.tolist()

        with ThreadPoolExecutor() as executor:
            executor.map(assign_embedding, documents, embeddings)
    
        print("embedding complete, computing similarities")
        
        if similarity_matrix:
            similarities = model.similarity(embeddings,embeddings)
            return documents,similarities
        else:
            return documents
