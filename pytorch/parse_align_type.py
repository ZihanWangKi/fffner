import pickle
import sys
import json

import numpy as np
from util import get_entity_info, get_p_r_f, softmax1d as softmax
import yaml


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


def evaluate(dir, dev=False):
    hparams = yaml.safe_load(open(f"{dir}/hparams.yaml"))
    label_to_entity_type_index, entity_type_names = get_entity_info(hparams["dataset_name"])

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

    if dev:
        predictions = pickle.load(open(f"{dir}/dev_predictions.pk", "rb"))
        outfile = open(f"{dir}/dev_type.txt", "w")
        outjson = open(f"{dir}/dev_type.json", "w")
    else:
        predictions = pickle.load(open(f"{dir}/predictions.pk", "rb"))
        outfile = open(f"{dir}/type.txt", "w")
        outjson = open(f"{dir}/type.json", "w")

    per_sid_results = predictions["per_sid_results"]
    tokens = predictions["tokens"]
    labels = predictions["labels"]
    gt = []
    p1 = []
    p2 = []
    for key in sorted(list(per_sid_results.keys())):
        outfile.write("{}\n".format("~" * 88))
        outfile.write("{}\n".format(tokens[key]))
        results = per_sid_results[key]
        gt_entities = []
        predictied_entities = []
        prediction_confidence = {}
        prediction_confidence_type = {}
        for span_start, span_end, ground_truth, prediction, ground_truth_type, prediction_type in results:
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
    return raw["f1"] + resolved["f1"]
    # print(get_p_r_f([(a,b,c,d) for a,b,c,d in gt if b == c], p2))


if __name__ == '__main__':
    evaluate(sys.argv[1])
