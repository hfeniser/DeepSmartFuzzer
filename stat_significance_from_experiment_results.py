import os
import statistics as stat
from scipy.stats import t
import numpy as np
results = {}

deephunter_results = []
tensorfuzz_results = []
mcts_results = []
mcts_clustered_results = []

for path, subdirs, files in os.walk("experiments"):
    for f in files:
        if f.endswith(".txt"):
            file_path = os.path.join(path, f)
            if file_path.find("adversarial") == -1:
                # parse file
                file = open(file_path, mode="r")
                lines = file.readlines()
                cov_inc = float(lines[-1].split(": ")[-1])
                if file_path.find("deephunter") != -1:
                    deephunter_results.append(cov_inc)
                elif file_path.find("tensorfuzz") != -1:
                    tensorfuzz_results.append(cov_inc)
                elif file_path.find("mcts_clustered") != -1:
                    mcts_clustered_results.append(cov_inc)
                elif file_path.find("mcts") != -1:
                    mcts_results.append(cov_inc)

print("----------DeepHunter----------")
print("Number of Experiments: %d" % len(deephunter_results))
print("Mean: %f" % stat.mean(deephunter_results))
print("Variance: %f" % stat.variance(deephunter_results))
print()
print("----------Tensorfuzz----------")
print("Number of Experiments: %d" % len(tensorfuzz_results))
print("Mean: %f" % stat.mean(tensorfuzz_results))
print("Variance: %f" % stat.variance(tensorfuzz_results))
print()
print("----------MCTS_Clustered----------")
print("Number of Experiments: %d" % len(mcts_clustered_results))
print("Mean: %f" % stat.mean(mcts_clustered_results))
print("Variance: %f" % stat.variance(mcts_clustered_results))
print()
print("----------MCTS----------")
print("Number of Experiments: %d" % len(mcts_results))
print("Mean: %f" % stat.mean(mcts_results))
print("Variance: %f" % stat.variance(mcts_results))
print()
print("----------MCTS vs. DeepHunter----------")
# (Mean1-Mean2)/sqrt(Var1/(Nb of Observations1-1) + Var2/(Nb of Observations2-1))
mcts_deephunter_t = (stat.mean(mcts_results) - stat.mean(deephunter_results))/np.sqrt(stat.variance(mcts_results)/(len(mcts_results)-1) + stat.variance(deephunter_results)/(len(deephunter_results)-1))
print("T-statistics: %f" % mcts_deephunter_t)
print("P-value: %f" % (1- t.cdf(mcts_deephunter_t, len(mcts_results) + len(deephunter_results) - 2)))
print()
print("----------MCTS vs. TensorFuzz----------")
# (Mean1-Mean2)/sqrt(Var1/(Nb of Observations1-1) + Var2/(Nb of Observations2-1))
mcts_tensorfuzz_t = (stat.mean(mcts_results) - stat.mean(tensorfuzz_results))/np.sqrt(stat.variance(mcts_results)/(len(mcts_results)-1) + stat.variance(tensorfuzz_results)/(len(tensorfuzz_results)-1))
print("T-value: %f" % mcts_tensorfuzz_t)
print("P-value: %f" % (1- t.cdf(mcts_tensorfuzz_t, len(mcts_results) + len(tensorfuzz_results) - 2)))
print()