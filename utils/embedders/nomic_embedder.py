from sentence_transformers import SentenceTransformer
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from llama_index.core.schema import TextNode
from llama_index.core.schema import QueryBundle
from typing import List,Union,Dict
from huggingface_hub import login
import torch.nn.functional as F

class NomicEmbedder:

    def __init__(self, huggingface_token, cache_loc, model_save_loc,matryoshka_dim):
        self.huggingface_token = huggingface_token
        self.cache_loc = cache_loc
        self.model_save_loc = model_save_loc
        self.matryoshka_dim = matryoshka_dim

    def download_embedding_model(self,model_name:str):
    
        login(token=self.huggingface_token)
        embedding_model = SentenceTransformer(model_name,
                                            cache_folder=self.cache_loc,
                                            trust_remote_code=True
                                            )
        embedding_model.save(path=f'{self.model_save_loc}/{model_name}', safe_serialization=True)

        print(f'Done downloading {model_name} to {self.model_save_loc}')


    def get_embeddings(self, model_name: str,
                       documents,
                       chunk_type,
                       similarity_matrix:bool):
        
        if similarity_matrix:   
            chunks = [f"{chunk_type}: " + (doc.text if isinstance(doc, TextNode) else doc.query_str) for doc in list(documents.values())]
        else:
            chunks = [f"{chunk_type}: " + (doc.text if isinstance(doc, TextNode) else doc.query_str) for doc in documents]

        model_location = f'{self.model_save_loc}/{model_name}'

        if not os.path.isdir(model_location):
            print(f'{model_name} not found in : {model_location}')
            print(f'please download {model_name} using obj.download_embedding_model(model_name="{model_name}")')
            print()
            return
            
        model = SentenceTransformer(model_location, trust_remote_code=True)
        embeddings = model.encode(chunks, convert_to_tensor=True,show_progress_bar=True)
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
        embeddings = embeddings[:, :self.matryoshka_dim]
        embeddings = F.normalize(embeddings, p=2, dim=1)


        if similarity_matrix:

            for key,embedding in zip(documents.keys(),embeddings):
                documents[key].embedding = embedding.tolist()

            similarities = model.similarity(embeddings,embeddings)

            print("embedding complete")
            return documents,similarities
        
        else:

            def assign_embedding(doc, embedding):
                doc.embedding = embedding.tolist()

            with ThreadPoolExecutor() as executor:
                executor.map(assign_embedding, documents, embeddings)

            print("embedding complete")
            return documents

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
