import pandas
from utils.evaluation_metrics._order_unaware import *
from utils.evaluation_metrics._order_aware import *

class RetrieverEvaluator:

    def __init__(self, df,col):

        self.df = df
        self.y = df['actual_contexts'].tolist()
        self.y_pred = df[col].tolist()

    def get_order_unaware_metrics(self,k):
        precision = []
        recall = []
        f1 = []

        for y,y_pred in zip(self.y,self.y_pred):

            results = order_unaware_metrics(y,y_pred,k)
            precision.append(results[f'precision@{k}'])
            recall.append(results[f'recall@{k}'])
            f1.append(results[f'F1@{k}'])

        avg_precision =  sum(precision) / len(precision) if len(precision) != 0 else 0
        avg_recall = sum(recall) / len(recall) if len(recall) != 0 else 0
        avg_f1 = sum(f1) / len(f1) if len(f1) != 0 else 0

        return {f'avg precision@{k}':avg_precision,
                f'avg recall@{k}':avg_recall,
                f'avg F1@{k}':avg_f1}
    
    def get_order_aware_metrics(self):
        mrr = []
        ndcg = []
        avg_precision = []

        for y,y_pred in zip(self.y,self.y_pred):

            results = order_aware_metrics(y,y_pred)
            mrr.append(results[f'mrr'])
            ndcg.append(results[f'ndcg'])
            avg_precision.append(results[f'avg_precision'])

        avg_mrr =  sum(mrr) / len(mrr) if len(mrr) != 0 else 0
        avg_ndcg = sum(ndcg) / len(ndcg) if len(ndcg) != 0 else 0
        mean_avg_precision = sum(avg_precision) / len(avg_precision) if len(avg_precision) != 0 else 0

        return {f'avg mrr':avg_mrr,
                f'avg ndcg':avg_ndcg,
                f'mean avg precision':mean_avg_precision}
    