
from typing import List

def precision_at_k(y, y_hat, k):

    boolean_list = [element in y for element in y_hat]
    relevent_k = sum(boolean_list[:k]) # No. of matches at k
    recall_k = relevent_k / k
    
    return recall_k

def recall_at_k(y, y_hat, k):

    boolean_list = [element in y for element in y_hat]
    relevent_k = sum(boolean_list[:k]) # No. of matches at k
    relevent_total = sum(boolean_list)
    recall_k = relevent_k / relevent_total if relevent_total != 0 else 0
    
    return recall_k

def f1_at_k(y_true: List[int], y_pred: List[int], k: int) -> float:
    """
    Calculate F1@k
    """
    precision = precision_at_k(y_true, y_pred, k)
    recall = recall_at_k(y_true, y_pred, k)

    # Calculate F1@k
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) !=0 else 0

    return f1

def order_unaware_metrics(y,y_pred,k):

    if len(y_pred) == 0:
        return {f'precision@{k}':0,
                f'recall@{k}':0,
                f'F1@{k}':0}
    
    if k> len(y_pred):
        y_pred.extend([-1] * (k - len(y_pred)))
    
    recall_k = recall_at_k(y,y_pred,k)
    precision_k = precision_at_k(y,y_pred,k)
    f1_k = f1_at_k(y,y_pred,k)

    return {f'precision@{k}':precision_k,
            f'recall@{k}':recall_k,
            f'F1@{k}':f1_k}