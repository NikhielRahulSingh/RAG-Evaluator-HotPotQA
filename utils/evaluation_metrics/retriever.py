import pandas
from utils.evaluation_metrics._order_unaware import *
from utils.evaluation_metrics._order_aware import *

class RetrieverEvaluator:

    def __init__(self, y,y_pred):

        self.y = y
        self.y_pred = y_pred

    def get_order_unaware_metrics(self,k):
        precision = []
        recall = []

        for y,y_pred in zip(self.y,self.y_pred):

            results = order_unaware_metrics(y,y_pred,k)
            precision.append(results[f'precision@{k}'])
            recall.append(results[f'recall@{k}'])

        avg_precision =  sum(precision) / len(precision) if len(precision) != 0 else 0
        avg_recall = sum(recall) / len(recall) if len(recall) != 0 else 0

        return {f'avg precision@{k}':avg_precision,
                f'avg recall@{k}':avg_recall,}
    
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
    