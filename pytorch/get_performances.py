import json
import numpy as np

ds = ["atis", "conll2003", "i2b2", "mitmovie", "multiwoz", "onto", "restaurant", "snips", "wikigold", "WNUT17"]
final_performance = []
for i in range(10):
    results = []
    for j in range(10):
        result = json.load(open(f"lightning_logs/version_{i * 10 + j}/results.json"))["resolved_f1"]
        results.append(result * 100)
    print(ds[i], np.mean(results), np.std(results))
    final_performance.append(np.mean(results))
print(np.mean(final_performance))