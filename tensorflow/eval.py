import pickle
import sys
import json

import numpy as np
import yaml
from collections import defaultdict
from create_data import NERDataset
def get_entity_info():
    config = {
        "PER": "Person",
        "LOC": "Location",
        "ORG": "Organization",
        "MISC": "Miscellaneous"
    }
    label_to_entity_type_index = {k: i for i, k in enumerate(list(config.keys()))}
    entity_type_names = list(config.values())
    return label_to_entity_type_index, entity_type_names

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


def softmax(x):
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def resolve(length, spans, prediction_confidence):
    used = [False] * length
    spans = sorted(spans, key=lambda x: prediction_confidence[(x[0], x[1])], reverse=True)
    real_spans = []
    for span_start, span_end, ent_type in spans:
        fill = False
        for s in range(span_start, span_end + 1):
            if used[s]:
                fill = True
                break
        if not fill:
            real_spans.append((span_start, span_end, ent_type))
            for s in range(span_start, span_end + 1):
                used[s] = True
    return real_spans


def evaluate(outputs, base):
    label_to_entity_type_index, entity_type_names = get_entity_info()

    def print_set(span_sets, tokens, prediction_confidence=None):
        out = []
        for start, end, ent_type in list(span_sets):
            text = " ".join(tokens[start: end + 1])
            o = (text,)
            if prediction_confidence is not None and (start, end) in prediction_confidence:
                o += (prediction_confidence[(start, end)],)
            o += (entity_type_names[ent_type],)
            out.append(o)
        return out

    outfile = open(f"type.txt", "w")
    outjson = open(f"type.json", "w")
    per_sid_results = defaultdict(list)
    is_entity_labels = outputs["labels_1"]
    entity_type_labels = outputs["labels_2"]
    is_entity_logits = outputs["preds_1"]
    entity_type_logits = outputs["preds_2"]
    st_ids = outputs["ids"]

    ids = list(range(1, len(is_entity_logits) + 1))
    print(len(is_entity_labels), len(entity_type_labels), len(is_entity_logits), len(entity_type_logits), len(base.id_to_sentence_infos))
    for id, stid, is_entity_label, is_entity_logit, entity_type_label, entity_type_logit in zip(ids,
                                                                                          st_ids,
                                                                                          is_entity_labels,
                                                                                          is_entity_logits,
                                                                                          entity_type_labels,
                                                                                          entity_type_logits):
        assert id == stid, f"{id} {stid}"
        if id > len(base.id_to_sentence_infos): break
        info = base.id_to_sentence_infos[id]
        per_sid_results[info["sid"]].append((info["span_start"], info["span_end"],
                                             is_entity_label, is_entity_logit, entity_type_label,
                                             entity_type_logit))
    tokens = base.all_tokens
    labels = base.all_labels


    gt = []
    p1 = []
    p2 = []
    a_1 = 0
    w_1 = 0
    a_2 = 0
    w_2 = 0
    for key in sorted(list(per_sid_results.keys())):
        outfile.write("{}\n".format("~" * 88))
        outfile.write("{}\n".format(tokens[key]))
        results = per_sid_results[key]
        # if len(results) != len(tokens[key]) * (len(tokens[key]) + 1) // 2:
        #     print(f"{len(results)} {len(tokens[key])}")
        #     print(key)
        #     for span_start, span_end, ground_truth, prediction, ground_truth_type, prediction_type in results:
        #         print(span_start, span_end)
        #     exit()
        gt_entities = []
        predictied_entities = []
        prediction_confidence = {}
        prediction_confidence_type = {}
        for span_start, span_end, ground_truth, prediction, ground_truth_type, prediction_type in results:
            # print(ground_truth, prediction, ground_truth_type, prediction_type)
            if np.argmax(prediction) == ground_truth:
                a_1 += 1
            else:
                w_1 += 1
            if np.argmax(prediction_type) == ground_truth_type:
                a_2 += 1
            else:
                w_2 += 1
            if ground_truth == 1:
                gt_entities.append((span_start, span_end, ground_truth_type))
            if prediction[1] > prediction[0]:
                predictied_entities.append((span_start, span_end, np.argmax(prediction_type).item()))
            prediction_confidence[(span_start, span_end)] = max(softmax(prediction))
            prediction_confidence_type[(span_start, span_end)] = max(softmax(prediction_type))

        gt.extend([(key, *x) for x in gt_entities])
        p1.extend([(key, *x) for x in predictied_entities])
        resolved_predicted = resolve(len(tokens[key]), predictied_entities, prediction_confidence)
        p2.extend([(key, *x) for x in resolved_predicted])

        gt_entities = set(gt_entities)
        predictied_entities = set(predictied_entities)
        predictied_entities_resolved = set(resolved_predicted)
        # print("common", print_set(gt_entities & predictied_entities, tokens[key], prediction_confidence))
        # print("fail to find", print_set(gt_entities - predictied_entities, tokens[key], prediction_confidence))
        # print("extra find", print_set(predictied_entities - gt_entities, tokens[key], prediction_confidence))
        #
        # print("common", print_set(gt_entities & predictied_entities_resolved, tokens[key], prediction_confidence))
        # print("fail to find", print_set(gt_entities - predictied_entities_resolved, tokens[key], prediction_confidence))
        # print("extra find", print_set(predictied_entities_resolved - gt_entities, tokens[key], prediction_confidence))

        outfile.write(
            "common: {}".format(print_set(gt_entities & predictied_entities, tokens[key], prediction_confidence)))
        outfile.write("\n")
        outfile.write(
            "fail to find: {}".format(print_set(gt_entities - predictied_entities, tokens[key], prediction_confidence)))
        outfile.write("\n")
        outfile.write(
            "extra find: {}".format(print_set(predictied_entities - gt_entities, tokens[key], prediction_confidence)))
        outfile.write("\n")

        outfile.write(
            "common: {}".format(
                print_set(gt_entities & predictied_entities_resolved, tokens[key], prediction_confidence)))
        outfile.write("\n")
        outfile.write("fail to find: {}".format(
            print_set(gt_entities - predictied_entities_resolved, tokens[key], prediction_confidence)))
        outfile.write("\n")
        outfile.write("extra find: {}".format(
            print_set(predictied_entities_resolved - gt_entities, tokens[key], prediction_confidence)))
        outfile.write("\n")

    outfile.write("{}".format(get_p_r_f(gt, p1)))
    outfile.write("\n")
    outfile.write("{}".format(get_p_r_f(gt, p2)))
    outfile.close()

    json.dump({
        "unresolved": get_p_r_f(gt, p1),
        "resolved": get_p_r_f(gt, p2),
    }, outjson, indent=2)
    outjson.close()

    raw = get_p_r_f(gt, p1)
    resolved = get_p_r_f(gt, p2)
    print(raw)
    print(resolved)
    print(a_1, w_1, a_2, w_2)
    return raw["f1"] + resolved["f1"]
    # print(get_p_r_f([(a,b,c,d) for a,b,c,d in gt if b == c], p2))


if __name__ == '__main__':
    import pickle

    OUTPUT_FILE = "something_v3.pk"
    BASE_FILE = "eval.pk"

    outputs = pickle.load(open(OUTPUT_FILE, "rb"))
    base = pickle.load(open(BASE_FILE, "rb"))

    evaluate(outputs, base)