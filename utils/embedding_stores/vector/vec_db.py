import chromadb
from llama_index.core.schema import TextNode,NodeWithScore
from typing import List
from chromadb.config import Settings
import concurrent.futures
import tqdm
 
import os


def create_vector_db(db_name:str,
                      save_loc:str,
                      docs:List[TextNode]):

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    def process_doc(doc):
        documents.append(doc.text)
        embeddings.append(doc.embedding)
        metadatas.append({'caption': doc.metadata['caption']})
        ids.append(doc.id_)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_doc, doc) for doc in docs]
        progress_bar = tqdm.tqdm(desc="processing docs,embeddings and metadatas... ",total=len(futures))
        for future in concurrent.futures.as_completed(futures):
            progress_bar.update(1)

    chroma_client = chromadb.PersistentClient(path=save_loc, 
                                              settings=Settings(anonymized_telemetry=False))
    
    collection = chroma_client.create_collection(name=db_name,metadata={"hnsw:space": "cosine"})

    # Split data into batches and add them to the collection
    batch_size: int = 41666
    num_batches = len(documents) // batch_size + (len(documents) % batch_size != 0)
    with tqdm.tqdm(total=num_batches, desc="Adding batches to ChromaDB...") as batch_progress:
        for i in range(0, len(documents), batch_size):
            collection.add(documents=documents[i:i + batch_size],
                           embeddings=embeddings[i:i + batch_size],
                           metadatas=metadatas[i:i + batch_size],
                           ids=ids[i:i + batch_size])
            batch_progress.update(1)


    print(f"vector db created")
    return collection

def load_vector_db(db_name,save_loc):

    chroma_client = chromadb.PersistentClient(path=save_loc, settings=Settings(anonymized_telemetry=False))

    db = chroma_client.get_collection(name=db_name)

    print(f"Vector Database retrieved")
    return db

def create_or_load_vector_db(db_name:str,
                             save_loc:str,
                             docs:List[TextNode]):
    
    try:
        V = load_vector_db(db_name=db_name,
                           save_loc=save_loc)
        return V
    except Exception as e:
        print(e)
        V = create_vector_db(db_name=db_name,
                             save_loc=save_loc,
                             docs=docs)
        return V