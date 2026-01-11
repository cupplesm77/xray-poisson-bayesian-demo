# poisson_example.py

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
import arviz as az
import preliz as pz

old=False

if old:
    draws = 100_000
    observed_count0 = 10
    p_min, p_max, = 0, 80
    loc, scale = 0, 1
    def simulate_posterior_poisson(n_draws=draws,
                                   prior_min=p_min,
                                   prior_max=p_max,
                                   loc=0,
                                   scale=1,
                                   observed_count=observed_count0,
                                   seed=42,
                                   prior_distribution='uniform',
                                   ):
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

        # employ a seed to ensure reproducibility
        rng = np.random.default_rng(seed)

        # 1. Prior:
        if prior_distribution == 'uniform':
            prior_samples = rng.uniform(prior_min, prior_max, size=n_draws)
        elif prior_distribution == 'normal':
            prior_samples = rng.normal(loc=loc , scale=scale, size=n_draws)
        else:
            raise ValueError(f"Unsupported prior distribution: {prior_distribution}")

        # 2. Likelihood: simulate Poisson counts from each λ
        simulated_counts = rng.poisson(lam=prior_samples)

        # 3. Posterior: keep λ values that produced the observed count
        posterior_samples = prior_samples[simulated_counts == observed_count]

        return prior_samples, posterior_samples
# else:
print("Running the New Routine")
def simulate_posterior_poisson(
        n_draws=100_000,
        observed_count=10,
        prior_distribution="uniform",
        prior_min=0.0,
        prior_max=80.0,
        loc=0.0,
        scale=1.0,
        seed=42,
):
    """
    Simulate a posterior distribution for a Poisson rate parameter λ
    using generative conditioning (likelihood-based rejection sampling).

    Parameters
    ----------
    n_draws : int, optional
        Number of prior samples to draw.
    observed_count : int
        Observed Poisson count.
    prior_distribution : {"uniform", "normal"}
        Choice of prior distribution for λ.
    prior_min : float, optional
        Lower bound for Uniform prior.
    prior_max : float, optional
        Upper bound for Uniform prior.
    loc : float, optional
        Mean of Normal prior.
    scale : float, optional
        Standard deviation of Normal prior.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    prior_samples : np.ndarray
        Samples drawn from the prior distribution.
    posterior_samples : np.ndarray
        Subset of prior samples whose simulated Poisson counts
        match the observed count.
    """

    rng = np.random.default_rng(seed)

    # --- Prior sampling -----------------------------------------------------
    if prior_distribution == "uniform":
        prior_samples = rng.uniform(prior_min, prior_max, size=n_draws)

    elif prior_distribution == "normal":
        prior_samples = rng.normal(loc=loc, scale=scale, size=n_draws)
        # enforce positivity for λ
        prior_samples = prior_samples[prior_samples > 0]

        # if too few samples remain, resample until we reach n_draws
        while len(prior_samples) < n_draws:
            extra = rng.normal(loc=loc, scale=scale, size=n_draws)
            extra = extra[extra > 0]
            prior_samples = np.concatenate([prior_samples, extra])
        prior_samples = prior_samples[:n_draws]

    else:
        raise ValueError(
            f"Unsupported prior distribution '{prior_distribution}'. "
            "Choose 'uniform' or 'normal'."
        )

    # --- Likelihood simulation ----------------------------------------------
    simulated_counts = rng.poisson(lam=prior_samples)

    # --- Posterior extraction -----------------------------------------------
    posterior_samples = prior_samples[simulated_counts == observed_count]

    return prior_samples, posterior_samples

# initalize a default prior
prev_posterior_poisson = None

observed_count0 = 10
def simulate_posterior_poisson_prior(n_draws=100_000,
                                     prior=prev_posterior_poisson,
                                     observed_count=observed_count0,
                                     seed=43,
                                     ):
    """
    Simulate a posterior distribution for a Poisson model using
    a previous Posterior and generative conditioning.

    Parameters
    ----------
    n_draws : int
        Number of prior samples to draw.
    prior :
        previous Posterior distribution.
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

    # logic for handling a None prior
    if prior is None:
        print("Prior is None, returning None")
        print("Please set an appropriate prior distribution.")
        return None, None

    # 1. Prior: If the prior is too small, resample it to n_draws
    if len(prior) < n_draws:
        prior_samples = rng.choice(prior, size=n_draws, replace=True)
        print(f"Resampled the Prior Distribution: Count={observed_count}.")
    else:
        prior_samples = prior

    # 2. Likelihood: simulate Poisson counts from each λ
    simulated_counts = rng.poisson(lam=prior_samples)

    # 3. Posterior: keep λ values that produced the observed count
    posterior_samples = prior_samples[simulated_counts == observed_count]

    return prior_samples, posterior_samples

# Plot a histogram
observed_count0 = 10
def plot_histograms(prior_samples,
                    posterior_samples,
                    observed_count=observed_count0,
                    bins=50,
                    ):
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


# plot the posterior density
def plot_posterior_density(
    idata,
    var_name,
    ci=0.94,
    color="steelblue",
    alpha=0.7,
    linewidth=2,
    facet=None,
    title=None,
):
    """
    Plot a posterior density using preLiz (Option 2).

    Parameters
    ----------
    idata : arviz.InferenceData or dict
        Posterior samples.
    var_name : str
        Name of the variable to plot.
    ci : float, optional
        Credible interval level (default 0.94).
    color : str, optional
        Line color.
    alpha : float, optional
        Transparency.
    linewidth : float, optional
        Line width.
    facet : str or None, optional
        Facet by "chain", "draw", or any group dimension.
    title : str or None, optional
        Optional title for the plot.
    """

    """
    Plot a posterior density using az.
    """

    # az.plot_posterior is the standard way to plot posterior distributions.
    # It handles numpy arrays and InferenceData objects seamlessly.
    ax = az.plot_posterior(
        idata,
        hdi_prob=ci,
        color=color,
        kind="kde",
        linewidth=linewidth,
        alpha=alpha,
    )

    if title:
        if hasattr(ax, 'set_title'):
            ax.set_title(title)
            ax.set_xlabel(var_name)
        else:
            # Handle cases where multiple axes might be returned (e.g. if idata is InferenceData)
            plt.suptitle(title)

    plt.show()
    return ax


if __name__ == "__main__":

    # example case for observed counts = 12
    obs_count = 12
    # Example usage
    prior, posterior = simulate_posterior_poisson(
        n_draws=100_000,
        prior_min=0,
        prior_max=80,
        observed_count=obs_count
    )

    plot_histograms(prior, posterior, observed_count=obs_count)

    # Show analytic comparison
    mean, (lo, hi) = analytic_posterior_gamma(observed_count=obs_count)
    print("Analytic posterior (Gamma) for comparison:")
    print(f"  Mean: {mean:.2f}")
    print(f"  95% interval: [{lo:.2f}, {hi:.2f}]")
    del mean, lo, hi, prior, posterior

    # example case for observed counts = 19
    obs_count = 19
    prior, posterior = simulate_posterior_poisson(
        n_draws=100_000,
        prior_min=0,
        prior_max=80,
        observed_count=obs_count
    )

    plot_histograms(prior, posterior, observed_count=obs_count)

    # Show analytic comparison
    mean, (lo, hi) = analytic_posterior_gamma(observed_count=obs_count)
    print("Analytic posterior (Gamma) for comparison:")
    print(f"  Mean: {mean:.2f}")
    print(f"  95% interval: [{lo:.2f}, {hi:.2f}]")
    del mean, lo, hi, prior, posterior


    # Now explore an example where we begin with an observation of 12, and then we subsequently observed 19
    # Use the posterior for 12 as a prior for the observed 19 case.


    # example case for observed counts = 12
    obs_count = 12
    # Example usage
    prior_12, posterior_12 = simulate_posterior_poisson(
        n_draws=100_000,
        prior_min=0,
        prior_max=80,
        observed_count=obs_count
    )

    # Re-sample from the posterior to restore sample size
    # This prevents the sample size from collapsing to near-zero in the subsequent step
    # rng = np.random.default_rng(42)
    # resampled_prior_for_19 = rng.choice(posterior_12, size=100_000, replace=True)

    obs_count = 19
    prior_19, posterior_19 = simulate_posterior_poisson_prior(prior=posterior_12,
                                                              observed_count=obs_count,
                                                              )
    n_bins = 30
    plot_histograms(prior_19,
                    posterior_19,
                    observed_count=obs_count,
                    bins=n_bins)


    # now using arv to visualize posteriors
    post12=posterior_12
    post19=posterior_19


    # posterior 12, uniform prior
    g = plot_posterior_density(
            post12,
            "lambda",
            ci=0.98,
            color="steelblue",
            alpha=0.7,
            linewidth=2,
            facet=None,
            title="Posterior Density for λ Given Observed Count = 12",
    )
    del g


    # posterior 19, uniform prior
    g = plot_posterior_density(
            post19,
            "lambda",
            ci=0.98,
            color="steelblue",
            alpha=0.7,
            linewidth=2,
            facet=None,
            title="Posterior Density for λ Given Observed Count = 19",
    )
    del g


    # Next assume an observation of 18 with the prior = posterior_19
    obs_count = 18
    prior_18, posterior_18 = simulate_posterior_poisson_prior(prior=post19,
                                                              observed_count=obs_count,
                                                              )

    n_bins = 30
    plot_histograms(prior_18,
                    posterior_18,
                    observed_count=obs_count,
                    bins=n_bins)

    # posterior 19
    g = plot_posterior_density(
            posterior_18,
            "lambda",
            ci=0.98,
            color="steelblue",
            alpha=0.7,
            linewidth=2,
            facet=None,
            title="Posterior Density for λ Given Observed Count = 18",
    )
    del g

    # Next assume an observation of 18 with the prior = posterior_19
    obs_count = 16
    prior_16, posterior_16 = simulate_posterior_poisson_prior(prior=posterior_18,
                                                              observed_count=obs_count,
                                                              )

    n_bins = 30
    plot_histograms(prior_16,
                    posterior_16,
                    observed_count=obs_count,
                    bins=n_bins)

    # posterior 16
    g = plot_posterior_density(
            posterior_16,
            "lambda",
            ci=0.98,
            color="steelblue",
            alpha=0.7,
            linewidth=2,
            facet=None,
            title="Posterior Density for λ Given Observed Count = 16",
    )
    del g
    del prior_12, posterior_12, prior_16, posterior_16, prior_18, posterior_18


    # Now Build the Inference Part of the Routine.