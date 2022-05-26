import json
import os
import numpy as np

PATHS = json.load(open("paths.json", "r"))
PATH_TO_DATA_DIR = PATHS["PATH_TO_DATA_DIR"]
PATH_TO_CONFIG_DIR = PATHS["PATH_TO_CONFIG_DIR"]


def get_p_r_f(truth, pred):
    n_pred = len(pred)
    n_truth = len(truth)
    n_correct = len(set(pred) & set(truth))
    precision = 1. * n_correct / n_pred if n_pred != 0 else 0.0
    recall = 1. * n_correct / n_truth if n_truth != 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0.0 else 0.0
    return {
        "n_pred": n_pred,
        "n_truth": n_truth,
        "n_correct": n_correct,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def softmax1d(x):
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_entity_info(dataset_name):
    with open(os.path.join(PATH_TO_CONFIG_DIR, f"entity_map_{dataset_name}.json")) as f:
        config = json.load(f)
    label_to_entity_type_index = {k: i for i, k in enumerate(list(config.keys()))}
    entity_type_names = list(config.values())
    return label_to_entity_type_index, entity_type_names
