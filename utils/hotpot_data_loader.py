import pandas as pd
from typing import List
from llama_index.core.schema import TextNode
from llama_index.core.schema import QueryBundle
import concurrent.futures
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
        df = pd.read_json("data/hotpot_train_v1.1.json")
        if self.SAMPLE != None :
            df_sampled =  df.sample(frac=1, random_state=42).reset_index(drop=True)
            df_sampled = df_sampled[df_sampled['level'] == self.DIFFICULTY]
            df = df_sampled.head(self.SAMPLE)

        self.df = df

    def set_fields(self):

        contexts = self.df['context'].tolist()
        answers = self.df['answer'].tolist()
        levels = self.df['level'].tolist()
        queries = self.df['question'].tolist()

        nodes = {}
        i:int = 0

        for refs,answer,level,query in zip(contexts,answers,levels,queries) :
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

    


