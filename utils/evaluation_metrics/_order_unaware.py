
from typing import List

def precision_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    true_positives = act_set & pred_set
    if len(pred_set) > 0:
        return round(len(true_positives) / float(len(pred_set)), 2)
    else:
        return 0.0

def recall_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    if len(pred_set)>0:
        return round(len(act_set & pred_set) / float(len(act_set)), 2)
    else:
        return 0.0


def order_unaware_metrics(y,y_pred,k):

    if len(y_pred) == 0:
        return {f'precision@{k}':0,
                f'recall@{k}':0,
                }
    
    if k> len(y_pred):
        y_pred.extend([-1] * (k - len(y_pred)))
    
    recall_k = recall_at_k(y,y_pred,k)
    precision_k = precision_at_k(y,y_pred,k)

    return {f'precision@{k}':precision_k,
            f'recall@{k}':recall_k,}