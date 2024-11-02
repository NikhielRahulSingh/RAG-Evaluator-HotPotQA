import pandas as pd
from typing import List
from llama_index.core.schema import TextNode
from llama_index.core.schema import QueryBundle
from langchain_core.documents import Document
from typing import List

class HotPotQA:

    def __init__(self,SAMPLE=None,DIFFICULTY=None,SEED=None):

        self.SAMPLE = SAMPLE
        self.DIFFICULTY = DIFFICULTY
        self.SEED = SEED

        self.benchmarks = {key: [] for key in ('level', 'question', 'answer', 'actual_contexts')}
        self.df = None

        self.questions:List[QueryBundle] = None
        self.contexts:List[TextNode] = None

        self.load_data()
        self.set_fields()

        
    def get_data(self):     

        return self.contexts, pd.DataFrame(self.benchmarks)

    def load_data(self):
        df = pd.read_json("data/hotpot_test_fullwiki_v1.json")
        if self.SAMPLE != None :
            df_sampled =  df.sample(frac=1, random_state=42).reset_index(drop=True)
            df = df_sampled.head(self.SAMPLE)

        self.df = df

    def set_fields(self):

        contexts = self.df['context'].tolist()
        answers = self.df['answer'].tolist()
        queries = self.df['question'].tolist()

        nodes = {}
        i:int = 0

        for refs,answer,level,query in zip(contexts,answers,queries) :
            benchmark_extract_ids = []
            for caption_extracts in refs:
                caption =caption_extracts[0] 
                sentence = " ".join(caption_extracts[1])
                sentence = ' '.join(sentence.split())
                node = TextNode(id_=str(i),text=sentence)
                node.metadata["caption"] = caption
                nodes[str(i)] = node
                benchmark_extract_ids.append(str(i))
                i+=1

            self.benchmarks['question'].append(QueryBundle(query_str=query))
            self.benchmarks['answer'].append(answer)
            self.benchmarks['level'].append(level) 
            self.benchmarks['actual_contexts'].append(benchmark_extract_ids)

        self.contexts = nodes

def load_test_data():  
    df = pd.read_json("data/hotpot_test_fullwiki_v1.json")#data\hotpot_test_fullwiki_v1.json
    questions = df["question"].tolist()
    contexts = df["context"].tolist()

    context_docs:List[Document] = []
    benchmarks:List[List[int]] = []
    context_id:int = 0

    for context in contexts:
        benchmark_context_ids:List[int] = []

        for title_and_sentences in context:
            title = title_and_sentences[0]
            benchmark_context_ids.append(context_id)

            sentences = " ".join(title_and_sentences[1])
            sentence = ' '.join(sentences.split())

            document:Document = Document(page_content=sentence,metadata={"ID":context_id,"Title":title})
            context_docs.append(document)

            context_id += 1

        benchmarks.append(benchmark_context_ids)
    
    return questions,context_docs,benchmarks



