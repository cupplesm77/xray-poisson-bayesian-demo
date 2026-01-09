"""
Bayesian Data Analysis Demo: Poisson Likelihood via Generative Simulation
-------------------------------------------------------------------------

This module demonstrates the core Bayesian idea using a Poisson model:

    1. Choose a prior over the Poisson rate parameter λ
    2. Simulate data from the generative model
    3. Condition on the observed data (keep only λ values that could have
       produced the observed count)
    4. Visualize the resulting posterior distribution

This is the same logic taught in introductory Bayesian courses:
Bayesian inference = prior × likelihood → posterior,
implemented here through simulation (Approximate Bayesian Computation style).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma


def simulate_posterior_poisson(n_draws=100_000, prior_min=0, prior_max=80,
                               observed_count=19, seed=42):
    """
    Simulate a posterior distribution for a Poisson model using
    a uniform prior and generative conditioning.

    Parameters
    ----------
    n_draws : int
        Number of prior samples to draw.
    prior_min : float
        Lower bound of Uniform prior for λ.
    prior_max : float
        Upper bound of Uniform prior for λ.
    observed_count : int
        Observed Poisson count (e.g., 19 clicks).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    prior_samples : np.ndarray
        Samples of λ from the prior.
    posterior_samples : np.ndarray
        Samples of λ that generated the observed count.
    """

    rng = np.random.default_rng(seed)

    # 1. Prior: Uniform(0, 80)
    prior_samples = rng.uniform(prior_min, prior_max, size=n_draws)

    # 2. Likelihood: simulate Poisson counts from each λ
    simulated_counts = rng.poisson(lam=prior_samples)

    # 3. Posterior: keep λ values that produced the observed count
    posterior_samples = prior_samples[simulated_counts == observed_count]

    return prior_samples, posterior_samples


def plot_histograms(prior_samples, posterior_samples,
                    observed_count=19, bins=50):
    """
    Plot prior and posterior histograms for visual comparison.
    """

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].hist(prior_samples, bins=bins, color="gray", alpha=0.7)
    ax[0].set_title("Prior Distribution for λ")
    ax[0].set_xlabel("λ (mean clicks per day)")
    ax[0].set_ylabel("Frequency")

    ax[1].hist(posterior_samples, bins=bins, color="steelblue", alpha=0.8)
    ax[1].set_title(f"Posterior Distribution for λ | y = {observed_count}")
    ax[1].set_xlabel("λ (mean clicks per day)")
    ax[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def analytic_posterior_gamma(observed_count, prior_min=0, prior_max=80):
    """
    For reference: if the prior were flat on [0, ∞), the posterior would be:

        λ | y ~ Gamma(shape = y + 1, rate = 1)

    This function returns the mean and 95% interval of that analytic posterior.
    """

    shape = observed_count + 1
    rate = 1.0

    mean = shape / rate
    lower = gamma.ppf(0.025, a=shape, scale=1/rate)
    upper = gamma.ppf(0.975, a=shape, scale=1/rate)

    return mean, (lower, upper)


if __name__ == "__main__":
    # Example usage
    prior, posterior = simulate_posterior_poisson(
        n_draws=100_000,
        prior_min=0,
        prior_max=80,
        observed_count=19
    )

    plot_histograms(prior, posterior, observed_count=19)

    # Show analytic comparison
    mean, (lo, hi) = analytic_posterior_gamma(observed_count=19)
    print("Analytic posterior (Gamma) for comparison:")
    print(f"  Mean: {mean:.2f}")
    print(f"  95% interval: [{lo:.2f}, {hi:.2f}]")
