import os
import statistics as stat
results = {}

for path, subdirs, files in os.walk("experiments"):
    for f in files:
        if f.endswith(".txt"):
            file_path = os.path.join(path, f)
            # parse file
            file = open(file_path, mode="r")
            last_line = file.readlines()[-1]
            cov_inc = float(last_line.split(": ")[-1])
            exp_name = file_path[:-6]
            if exp_name not in results:
                results[exp_name] = [cov_inc]
            else:
                results[exp_name].append(cov_inc)
result_str = []
for exp_name in results:
    m = stat.mean(results[exp_name])
    std = stat.stdev(results[exp_name])
    result_str.append("%s: %0.2f ± %0.2f"%(exp_name, m, std))

result_str.sort()
print("\n".join(result_str))