import pickle
import sys
import json

import numpy as np
from util import get_p_r_f, softmax1d as softmax
import yaml


def resolve(length, spans, prediction_confidence):
    used = [False] * length
    spans = sorted(spans, key=lambda x: prediction_confidence[x], reverse=True)
    real_spans = []
    for span_start, span_end in spans:
        fill = False
        for s in range(span_start, span_end + 1):
            if used[s]:
                fill = True
                break
        if not fill:
            real_spans.append((span_start, span_end))
            for s in range(span_start, span_end + 1):
                used[s] = True
    return real_spans


def evaluate(dir, dev=False):
    def print_set(span_sets, tokens, prediction_confidence=None):
        out = []
        for start, end in list(span_sets):
            text = " ".join(tokens[start: end + 1])
            if prediction_confidence is not None and (start, end) in prediction_confidence:
                out.append((text, prediction_confidence[(start, end)]))
            else:
                out.append(text)
        return out

    if dev:
        predictions = pickle.load(open(f"{dir}/dev_predictions.pk", "rb"))
        outfile = open(f"{dir}/dev_cls.txt", "w")
        outjson = open(f"{dir}/dev_cls.json", "w")
    else:
        predictions = pickle.load(open(f"{dir}/predictions.pk", "rb"))
        outfile = open(f"{dir}/cls.txt", "w")
        outjson = open(f"{dir}/cls.json", "w")

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
        for span_start, span_end, ground_truth, prediction, _, _ in results:
            if ground_truth == 1:
                gt_entities.append((span_start, span_end))
            if prediction[1] > prediction[0]:
                predictied_entities.append((span_start, span_end))
            prediction_confidence[(span_start, span_end)] = max(softmax(prediction))

        gt.extend([(key, *x) for x in gt_entities])
        p1.extend([(key, *x) for x in predictied_entities])
        resolved_predicted = resolve(len(tokens[key]), predictied_entities, prediction_confidence)
        p2.extend([(key, *x) for x in resolved_predicted])

        gt_entities = set(gt_entities)
        predictied_entities = set(predictied_entities)
        predictied_entities_resolved = set(resolved_predicted)
        outfile.write(
            "common: {}".format(print_set(gt_entities & predictied_entities, tokens[key], prediction_confidence)))
        outfile.write("\n")
        outfile.write(
            "fail to find: {}".format(print_set(gt_entities - predictied_entities, tokens[key], prediction_confidence)))
        outfile.write("\n")
        outfile.write(
            "extra find: {}".format(print_set(predictied_entities - gt_entities, tokens[key], prediction_confidence)))
        outfile.write("\n")

        outfile.write("common: {}".format(
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


if __name__ == '__main__':
    evaluate(sys.argv[1])
