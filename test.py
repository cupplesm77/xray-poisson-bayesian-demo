import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import MatplotlibDeprecationWarning
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import seaborn as sns
import scipy.stats as stats
N = 5
size = 5
rng = np.random.default_rng(42)

l = np.array([200, 2000, 20000, 200000, 2000000])

# Subtracting the maximum shifts all log weights so the largest one becomes 0.

# Then exponentiating gives values between 0 and 1, instead of astronomically large or tiny numbers.

# Normalization (dividing by the sum) restores the correct relative weights.

# This trick is standard in numerical probability work — often called the log‑sum‑exp trick.

llog = np.log(l)

print(f"Original list: {l}")
print(f"Log-transformed list: {llog}")

llog -= np.max(llog)                # center to avoid overflow

print(f"Log centered weights: {llog}")

w = np.exp(llog)

print(f"Centered weights: {w}")

w /= w.sum()

print(f"Normalized weights: {w}")

# This next line mimics drawing samples from a Posterior
# distribution in Bayesian analysis:

# p=w: This argument tells the generator to pick indices
# based on the calculated weights.
# Result: Since the last element has the highest weight,
# the resulting idx array will likely contain mostly the
# index 4 (corresponding to 2,000,000).

idx = rng.choice(N, size=N, replace=True, p=w)

print("idx: ", idx)

p_post_samples = l[idx]

print(f"Post samples: {p_post_samples}")