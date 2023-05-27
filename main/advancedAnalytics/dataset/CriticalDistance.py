import numpy as np
from scipy import stats
from sklearn import metrics

scores = np.array([
    [0.875, 0.874, 0.869,0.863],
    [0.915, 0.913, 0.925,0.897],
    [0.869, 0.911, 0.923,0.907]
])

scores = np.array([0.875, 0.874, 0.869,0.863])

# Perform the Friedman test
friedman_result = stats.friedmanchisquare(*scores.T)
print("Friedman test p-value:", friedman_result.pvalue)

# Perform pairwise t-tests
n_datasets, n_classifiers = scores.shape
pairwise_results = np.zeros((n_classifiers, n_classifiers))

for i in range(n_classifiers):
    for j in range(i + 1, n_classifiers):
        t, p = stats.ttest_rel(scores[:, i], scores[:, j])
        pairwise_results[i, j] = p
        print(pairwise_results)

# Calculate critical distance
alpha = 0.05
k = n_classifiers
N = n_datasets
CD = stats.rankdata(-scores, axis=1).mean(axis=0)
CD *= stats.norm.ppf(1 - alpha / (2 * k * (k + 1) / (6 * N))**0.5)
print(CD)
# Rank-based comparisons
model1 = 0  # Index of Model 1
model2 = 1  # Index of Model 2

sorted_indices = np.argsort(pairwise_results[model1])
alpha_index = np.where(sorted_indices == model2)[0][0]
diff_rank = abs(CD[model1] - CD[model2])
print(diff_rank)
if diff_rank > CD[alpha_index]:
    print("Model 1 and Model 2 have significantly different performances.")
else:
    print("No significant difference between Model 1 and Model 2.")