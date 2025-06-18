import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import glob
import os

csv_files = glob.glob(os.path.join("results", "results_*.csv"))

df_list = [pd.read_csv(f) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)

correlations = {
    "cpu_utilization": [],
    "memory_utilization": [],
    "rt": []
}
# df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
# df['timestamp_10min'] = df['timestamp'].dt.floor('10h')

# agg_df = df.groupby(['timestamp_10min', 'msinstanceid'])[
#     ['cpu_utilization', 'memory_utilization', 'rt', 'mcr']
# ].mean().reset_index()

# print(agg_df.head())

# grouped = agg_df.groupby("msinstanceid")
grouped = df.groupby("msinstanceid")
print(grouped.head())
for ms, group in grouped:
    # print(ms, group)
    # if group.shape[0] < 500:
        if len(group) >= 2:
            for col in correlations:
                corr, _ = spearmanr(group["mcr"], group[col])
                if not np.isnan(corr):
                    correlations[col].append(corr)

plt.figure(figsize=(8, 6))
for metric, values in correlations.items():
    sorted_corr = np.sort(values)
    cdf = np.arange(1, len(sorted_corr)+1) / len(sorted_corr)
    plt.plot(sorted_corr, cdf, label=metric.replace("_", " ").title())

plt.axvline(0.6, color='red', linestyle='--', linewidth=1.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=1)

plt.xlabel("Spearman Correlation with MCR")
plt.ylabel("CDF")
plt.title("CDF of Spearman Correlation per Metric vs MCR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
