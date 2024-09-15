from llama_index.core.evaluation.retrieval import metrics

def order_aware_metrics(y,y_hat,):

    if len(y_hat) == 0:
        return {'mrr':0.0,
                'ndcg':0.0,
                'avg_precision':0.0}
    
    mrr = metrics.MRR(use_granular_mrr=True).compute(expected_ids=y,retrieved_ids=y_hat).score
    ndcg = metrics.NDCG().compute(expected_ids=y,retrieved_ids=y_hat).score
    avg_p = metrics.AveragePrecision().compute(expected_ids=y,retrieved_ids=y_hat).score

    return {'mrr':mrr,
            'ndcg':ndcg,
            'avg_precision':avg_p}