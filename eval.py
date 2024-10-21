from collections import Counter
import multiprocessing

from scipy.optimize import linear_sum_assignment
import numpy as np

from tqdm import tqdm
from chainlite import get_logger
logger=get_logger(__name__)

def tuple_superset(
    predicted_results: list, gold_results: list, discard_label_description=True
):
    if predicted_results is None:
        return gold_results is None
    if type(predicted_results) is bool or type(gold_results) is bool:
        return predicted_results == gold_results

    # each gold result should appear somewhere in predicted results,
    # neglecting the keys (which are the variable names)
    for gold_result in gold_results:
        if discard_label_description:
            gold_values = [
                (j[0], j[1])
                for i in gold_result
                for j in gold_result[i].items()
                if not i.endswith("Label") or i.endswith("Description")
            ]
        else:
            gold_values = [
                (j[0], j[1]) for i in gold_result.values() for j in i.items()
            ]

        found = False
        # loop through predicted to try to find the sublist
        for predicted_result in predicted_results:
            predicted_values = [
                (j[0], j[1]) for i in predicted_result.values() for j in i.items()
            ]

            if set(gold_values) <= set(predicted_values):
                found = True
                break

        if not found:
            return False

    return True


def _compute_match_ratio(predicted, gold):
    """
    Example `predicted` or `gold`:
    {
        "item": {
          "type": "uri",
          "value": "http://www.wikidata.org/entity/Q98926"
        },
        "itemLabel": {
          "type": "literal",
          "value": "Lola Landau",
          "xml:lang": "en"
        }
    }
    """
    gold_values = [
        tuple(sorted(gold_value.items()))
        for gold_key, gold_value in gold.items()
        if not (gold_key.endswith("Label") or gold_key.endswith("Description"))
    ]
    useful_predicted_values = [
        tuple(sorted(predicted_value.items()))
        for predicted_value in predicted.values()
        if tuple(sorted(predicted_value.items())) in gold_values
    ]

    # Find the intersection (minimum counts)
    overlap = sum((Counter(gold_values) & Counter(useful_predicted_values)).values())
    if len(gold_values) == 0:
        logger.error("zero gold values for %s\n\n%s", predicted, gold)
        return 0

    return overlap / len(gold_values)

def f1_simple(predicted_results: list, gold_results: list):
    """
    Simple F1 (no consideration for wikidata object results)
    """
    
    def safe_divide(x, y):
        if x == 0 and y == 0:
            return 0
        return x / y

    true_positive = [x for x in predicted_results if x in gold_results]
    false_positive = [x for x in predicted_results if x not in gold_results]
    false_negative = [x for x in gold_results if x not in predicted_results]

    precision = safe_divide(len(true_positive), len(true_positive) + len(false_positive))
    recall    = safe_divide(len(true_positive), len(true_positive) + len(false_negative))
    if precision + recall == 0:
        this_f1 = 0
    else:
        this_f1 = 2 * precision * recall / (precision + recall)
        
    return this_f1

def f1(predicted_results, gold_results, maximal_matching=True):
    """
    Calculates a row-major F1 score for each example.
    """
    if predicted_results is None:
        return 0

    if type(predicted_results) is bool or type(gold_results) is bool:
        return int(predicted_results == gold_results)

    if maximal_matching:
        # first compute a cost matrix between `predicted_results` and `gold_results`
        cost_matrix = np.empty((len(predicted_results), len(gold_results)))
        
        for i in range(len(predicted_results)):
            for j in range(len(gold_results)):
                i_j_recall = _compute_match_ratio(predicted_results[i], gold_results[j])
                cost_matrix[i , j] = i_j_recall
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
        
        assigned_values = cost_matrix[row_ind, col_ind]

        # true positives are those that get matched (times their respective row-by-row recall)
        tp = assigned_values[assigned_values > 0].sum()

        # each matched row below 1 but above 0 will count as 1 - recall(i,j) to false negatives
        below_one_above_zero = assigned_values[(assigned_values < 1) & (assigned_values > 0)]
        fp_or_fn = (1 - below_one_above_zero).sum()
        
        # each matched row PAIR with 0 match rate will count as 1 false negative and 1 false positive
        fp_or_fn += 2 * np.sum(assigned_values <= 0)

        # each individual unmatched row (due to cost matrix being rectangular) will
        # count as either 1 false negative or 1 false positive
        fp_or_fn += (len(predicted_results) - len(row_ind)) + (len(gold_results) - len(col_ind))
        
        res = 2 * tp / (2 * tp + fp_or_fn)

    else:
        # an older implmentation with greedy matching
        # SHOULD NO LONGER BE USED
        gold_result_mapping = [
            [gold_result, False]
            for gold_result in gold_results  # false denoting not matched yet
        ]

        tp = 0
        fp = 0
        fn = 0

        for predicted_result in predicted_results:
            candidate_index = None
            match_ratio = 0

            # go over gold results yet to be matched to find the one with most matches
            # greedily match that to this
            for index, gold_result in enumerate(gold_result_mapping):
                if gold_result[1] == True:
                    continue

                gold_result = gold_result[0]
                this_match_ratio = _compute_match_ratio(predicted_result, gold_result)
                if this_match_ratio > match_ratio:
                    match_ratio = this_match_ratio
                    candidate_index = index

            if candidate_index is not None:
                gold_result_mapping[candidate_index][1] = True

            if match_ratio == 0:
                fp += 1
            else:
                tp += match_ratio
                fn += 1 - match_ratio

        fn += len(list(filter(lambda x: x[1] == False, gold_result_mapping)))

        res = 2 * tp / (2 * tp + fp + fn)
    
    assert(0 <= res)
    assert(res <= 1)
    
    return res

def f1_wrapper(t):
    predicted_results, gold_results = t
    return f1(predicted_results, gold_results)

def parallel_f1(predicted_results_list, gold_results_list):
    """
    Runs the f1 function in parallel using multiprocessing.

    :param predicted_results_list: List of lists of predicted results
    :param gold_results_list: List of lists of gold results
    :return: List of F1 scores for each pair of predicted and gold results
    """
    if len(predicted_results_list) != len(gold_results_list):
        raise ValueError(
            "The length of predicted_results_list and gold_results_list must be the same."
        )

    with multiprocessing.Pool(16) as pool:
        f1_scores = list(tqdm(pool.imap(f1_wrapper, zip(predicted_results_list, gold_results_list)), total=len(predicted_results_list), desc="Calculating F1"))


    return f1_scores


if __name__ == "__main__":
    # more unit tests in `test_eval.py`

    predicted = [
        {
            "item": {"type": "uri", "value": "b"},
            "surprise": {"type": "uri", "value": "c"},
        },
        {
            "item": {"type": "uri", "value": "a"},
            "surprise": {"type": "uri", "value": "c"},
        },
        {
            "item": {"type": "uri", "value": "surprise"},
            "surprise": {"type": "uri", "value": "c"},
        },
        {
            "item": {"type": "uri", "value": "surprise_2"},
            "surprise": {"type": "uri", "value": "c"},
        },
    ]
    gold = [
        {
            "item": {"type": "uri", "value": "b"},
            "surprise": {"type": "uri", "value": "c"},
            "surprise": {"type": "uri", "value": "c"},
        },
        {
            "item": {"type": "uri", "value": "a"},
            "surprise": {"type": "uri", "value": "c"},
            "surprise": {"type": "uri", "value": "c"},
        },
    ]
    f1(predicted, gold)
