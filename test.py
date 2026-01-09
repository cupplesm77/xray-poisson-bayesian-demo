import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import MatplotlibDeprecationWarning
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import seaborn as sns
import scipy.stats as stats



N = 100000
size = 100000
rng = np.random.default_rng(42)


l = [1, 200000000]
# from this l list, create a long list of numbers between 1 and 200000000
long_list = np.random.randint(l[0], l[1], size=100_000)

larray = np.array(long_list)

# Subtracting the maximum shifts all log weights so the largest one becomes 0.

# Then exponentiating gives values between 0 and 1, instead of astronomically large or tiny numbers.

# Normalization (dividing by the sum) restores the correct relative weights.

# This trick is standard in numerical probability work — often called the log‑sum‑exp trick.

llog = np.log(larray)

print(f"Original list: {larray[:5]}")
print(f"Log-transformed list: {llog[:5]}")

llog -= np.max(llog)                # center to avoid overflow

print(f"Log centered weights: {llog}")

w = np.exp(llog)

print(f"Centered weights: {w}")

w /= w.sum()

print(f"Normalized weights: {w}")
print(f"size of weights: {w.shape}")

# This next line mimics drawing samples from a Posterior
# distribution in Bayesian analysis:

# p=w: This argument tells the generator to pick indices
# based on the calculated weights.
# Result: Since the last element has the highest weight,
# the resulting idx array will likely contain mostly

# Weighted sampling with replacement
idx = rng.choice(N, size=N, replace=True, p=w)
# This lines means:
# • 	It generates an array idx of length N.
# • 	Each entry is an integer between 0 and N-1.
# • 	The distribution of values follows the probability weights w.
# • 	Because , replace=True, some integers may appear multiple times, others may not appear at all.
# 🧠 Interpretation
# This is essentially weighted resampling with replacement.
# It’s often used in:
# - Bootstrap sampling (statistics, machine learning).
# - Particle filters (sensor fusion, Bayesian inference).
# - Monte Carlo simulations where you want to respect a probability distribution.


print("idx: ", idx)

p_post_samples = long_list[idx]

print(f"Post samples: {p_post_samples}")


l = [3164, 3362, 4435, 3542, 3578, 4529]
mean_l = np.mean(l)
sigma_l = np.std(l)

print(f"Mean: {mean_l}, Standard Deviation: {sigma_l}")
