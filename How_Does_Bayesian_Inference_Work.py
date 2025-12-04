# How_Does_Bayesian_Inference_Work.py

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import MatplotlibDeprecationWarning
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import seaborn as sns
import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


n_samples = 100000
n_ads_shown = 100
proportion_clicks = np.random.uniform(low=0.0, high=0.2, size=n_samples)
n_visitors = np.random.binomial(n=n_ads_shown , p = proportion_clicks, size=n_samples)

print("Type Proportion of Clicks:")
print(type(proportion_clicks))
print("Size Proportion of Clicks:")
print(len(proportion_clicks))
print()
print(proportion_clicks[:100])
print()

print("Type Number of visitors:")
print(type(n_visitors))
print("Size Number of visitors:")
print(len(n_visitors))
print()
print(n_visitors[:100])
print()

# Visualize proportion clicks
sns.histplot(proportion_clicks)
plt.xlabel('Proportion of visitors')
plt.show()

# Visualize n_visitors
sns.histplot(n_visitors, bins=20)
plt.xlabel('Number of visitors')
plt.show()

prior = pd.DataFrame(data={'proportion_clicks': proportion_clicks, 'n_visitors': n_visitors})

print(prior.head(100))

sns.jointplot(data=prior,
              x=n_visitors,
              y=proportion_clicks,
                )
plt.ylabel('Proportion of Clicks')
plt.xlabel('Number of visitors')
plt.title('Marginal Plot of Proportion of Visitors vs Number of Visitors')
plt.tight_layout()
plt.show()

# demonstrate correct simulation of the posterior for our problem:

# import numpy as np
# import scipy.stats as stats
# import seaborn as sns
# import matplotlib.pyplot as plt

# Data and prior
a, b = 5, 95
n, k = 100, 13
N = 100_000
rng = np.random.default_rng(42)

# 1) Prior distribution object
prior = stats.beta(a, b)

# 2) Candidate grid of p values
p_grid = np.linspace(0, 1, 500)

# 3) Prior density at each grid point
prior_density = prior.pdf(p_grid)

# 4) Likelihood at each grid point (Binomial, dropping constant C(n,k))
likelihood = (p_grid**k) * ((1 - p_grid)**(n - k))

# 5) Unnormalized posterior = prior × likelihood
unnormed_posterior = prior_density * likelihood

# 6) Normalize to make it a proper distribution
posterior = unnormed_posterior / np.trapezoid(unnormed_posterior, p_grid)

# Plot
plt.figure(figsize=(8,5))
sns.lineplot(x=p_grid, y=unnormed_posterior, label="Unnormalized posterior")
sns.lineplot(x=p_grid, y=posterior, label="Normalized posterior")
sns.lineplot(x=p_grid, y=prior_density, label="Prior", linestyle="--")
plt.legend()
plt.xlabel("p")
plt.ylabel("Probability Density")
plt.title("Prior × Likelihood → Posterior")
plt.show()


# Now demonstrate the exact solution using a beta-binomial model:
posterior = stats.beta(a + k, b + n - k)
plt.figure(figsize=(8,5))
sns.lineplot(x=p_grid, y=posterior.pdf(p_grid), label="Exact posterior")
sns.lineplot(x=p_grid, y=prior_density, label="Prior", linestyle="--")
plt.legend()
plt.xlabel("p")
plt.ylabel("Probability Density")
plt.title("Exact Posterior vs Prior")
plt.show()

# Use an approximate method to find the Expectation value of the posterior
# Don't use grids, but do use sampling of the distributions.
# Also, use the following:
# Take the logarithm of the likelihood to avoid numerical underflow or overflow
# Subtracting the maximum log-likelihood to shift all log weights so the largest one becomes 0.
# Then exponentiating gives values between 0 and 1, instead of astronomically large or tiny numbers.
# Normalization (dividing by the sum) restores the correct relative weights.
# This trick is standard in numerical probability work — often called the log‑sum‑exp trick.
# Now get the expectation value of the posterior.

# Sample from the prior
p_samples = prior.rvs(size=N, random_state=rng)

# Calculate log-likelihood: log(L) = k*log(p) + (n-k)*log(1-p)
log_likelihood = k * np.log(p_samples) + (n - k) * np.log(1 - p_samples)

# Apply log-sum-exp trick
# Subtract max log-likelihood to prevent overflow/underflow when exponentiating
log_likelihood_centered = log_likelihood - np.max(log_likelihood)
weights = np.exp(log_likelihood_centered)
weights_normalized = weights / np.sum(weights)

# Calculate expected value (weighted mean of the samples)
approx_expectation = np.sum(weights_normalized * p_samples)

print(f"Approximate Posterior Expectation: {approx_expectation:.5f}")
print(f"Exact Posterior Expectation:       {(a + k) / (a + b + n):.5f}")