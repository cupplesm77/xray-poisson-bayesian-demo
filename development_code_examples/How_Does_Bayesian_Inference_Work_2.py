# How_Does_Bayesian_Inference_Work_2.py


from multiprocessing.spawn import freeze_support

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import MatplotlibDeprecationWarning
# from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import seaborn as sns
import scipy.stats as stats
import pytensor
import pymc as pm
import arviz as az

# Set PyTensor to use Numba to avoid BLAS linking issues on Windows
# ---------------------------------------------------------------------------
# NOTE ON PYTENSOR BACKENDS (FAST_RUN vs NUMBA)
#
# PyTensor’s default execution mode (“FAST_RUN”) relies on BLAS libraries
# such as OpenBLAS or MKL. On Windows, especially when PyMC/PyTensor are
# installed via pip, BLAS often fails to link correctly, producing warnings
# and causing PyTensor to fall back to slower pure‑Python ops.
#
# The most reliable way to use FAST_RUN with full BLAS acceleration is to
# install PyMC and its dependencies through a conda environment, where BLAS
# libraries are bundled and properly configured.
#
# Since this project is running on Windows with a pip-based environment,
# NUMBA mode is selected to avoid BLAS linking issues and to provide stable,
# JIT‑compiled performance for PyMC sampling.
# ---------------------------------------------------------------------------
pytensor.config.mode = 'NUMBA'

import warnings
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

rng = np.random.default_rng(42)


def plot_grid_posterior(
        prior_dist,
        likelihood_fn,
        p_min=0.0,
        p_max=1.0,
        grid_size=500,
        title="Posterior from Grid Approximation",
        xlim=None,
        prior_label="Prior"
):
    """
    Compute and plot prior, likelihood, and posterior on a grid.

    Parameters
    ----------
    prior_dist : scipy.stats distribution object
        e.g., stats.uniform(loc=0, scale=0.2) or stats.beta(a, b)
    likelihood_fn : callable
        A function f(p_grid) that returns likelihood values for each p.
    p_min, p_max : float
        Range of p-grid
    grid_size : int
        Number of grid points
    title : str
        Plot title
    xlim : tuple or None
        Optional x-axis limits
    prior_label : str
        Label for the prior curve
    """

    # 1. Grid
    p_grid = np.linspace(p_min, p_max, grid_size)

    # 2. Prior density
    prior_density = prior_dist.pdf(p_grid)

    # 3. Likelihood (user-supplied function)
    likelihood = likelihood_fn(p_grid)

    # 4. Unnormalized posterior
    unnormed_posterior = prior_density * likelihood

    # 5. Normalize
    posterior = unnormed_posterior / np.trapezoid(unnormed_posterior, p_grid)

    # 6. Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(x=p_grid, y=unnormed_posterior, label="Unnormalized posterior", ax=ax)
    sns.lineplot(x=p_grid, y=posterior, label="Normalized posterior", ax=ax)
    sns.lineplot(x=p_grid, y=prior_density, label=prior_label, linestyle="--", ax=ax)

    ax.set_xlabel("probability of click, p")
    ax.set_ylabel("Probability Density")
    ax.set_title(title)
    ax.legend()

    if xlim:
        ax.set_xlim(*xlim)

    plt.show()
    return p_grid, posterior


# Various Likelihoods

# binomial Likelihood
def binomial_likelihood(n, k):
    return lambda p: (p ** k) * ((1 - p) ** (n - k))


# poisson Likelihood
def poisson_likelihood(y):
    return lambda lam: np.exp(-lam) * lam ** y


# normal Likelihood
def normal_likelihood(mu_obs, sigma):
    return lambda mu: stats.norm(mu, sigma).pdf(mu_obs)


# custom Likelihood
# def custom_likelihood(data):
#     return lambda theta: np.exp(-np.sum((data - model(theta)) ** 2))


def main():
    # """
    # The first set of code below is based on a joint distribution; you can see the marginal distributions in the plots
    # Further, we condition the data by choosing a particular number of visitors.  That led us to a distribution of
    # propotion of clicks that give us this number of visitors:   Question:
    #
    # Is the code below, "real" bayesian analysis?
    #
    # Short answer: **yes — this *is* Bayesian data analysis**, just at the most introductory, simulation‑based level.
    #
    # Let me unpack why, because this is one of those moments where the *method* feels almost too simple,
    # yet it’s genuinely Bayesian.
    #
    # ---
    #
    # # 🌟 Why this *is* Bayesian Data Analysis
    #
    # What you just did is the classic **generative‑model / prior‑predictive simulation** approach:
    #
    # 1. **Choose a prior**
    #    ```r
    #    mean_clicks <- runif(n_draws, min = 0, max = 80)
    #    ```
    #    This encodes your belief about the plausible range of daily clicks *before* seeing data.
    #
    # 2. **Generate data from the likelihood**
    #    ```r
    #    n_visitors <- rpois(n = n_draws, mean_clicks)
    #    ```
    #    This is the Poisson generative model:
    #    \[
    #    y \sim \text{Poisson}(\lambda)
    #    \]
    #
    # 3. **Condition on the observed data**
    #    ```r
    #    posterior <- prior[prior$n_visitors == 19, ]
    #    ```
    #    This is the key Bayesian step:
    #    You keep only the parameter values that *could have plausibly produced the observed data* (19 clicks).
    #
    # 4. **Visualize the posterior**
    #    ```r
    #    hist(posterior$mean_clicks)
    #    ```
    #
    # This is literally the **Approximate Bayesian Computation (ABC)** idea in its simplest form:
    #
    # > Simulate from the prior → simulate data → keep the parameter values that match the observed data.
    #
    # It’s crude, but it *is* Bayesian.
    #
    # ---
    #
    # # 🌟 Why it feels “too simple”
    #
    # Because you’re doing the Bayesian update by **filtering** instead of by:
    #
    # - computing the likelihood explicitly
    # - multiplying prior × likelihood
    # - normalizing
    #
    # But mathematically, it’s the same idea.
    #
    # This is the “generative intuition” that McElreath, Gelman, and many modern Bayesian teachers emphasize:
    #
    # > *Bayesian inference is just conditioning on data.*
    #
    # Your simulation is literally doing that.
    #
    # ---
    #
    # # 🌟 Why the histogram gives 12–28
    #
    # Your posterior distribution is:
    #
    # \[
    # p(\lambda \mid y=19)
    # \]
    #
    # Given:
    #
    # - prior: Uniform(0, 80)
    # - likelihood: Poisson(λ)
    #
    # The posterior is proportional to:
    #
    # \[
    # \lambda^{19} e^{-\lambda}
    # \]
    #
    # Which is a **Gamma(20, 1)** distribution (if the prior were flat on λ ≥ 0).
    #
    # The mean of that is 20, and the 95% interval is roughly 12–28.
    #
    # So your eyeballing was correct.
    #
    # ---
    #
    # # 🌟 Is this “real” Bayesian analysis?
    #
    # Yes — but it’s **introductory** Bayesian analysis.
    #
    # It’s meant to teach:
    #
    # - how priors generate predictions
    # - how likelihoods connect parameters to data
    # - how conditioning works
    # - how the posterior emerges from the generative model
    #
    # Later chapters will introduce:
    #
    # - analytic posteriors
    # - grid approximation
    # - MCMC
    # - hierarchical models
    # - model comparison
    #
    # But the foundation is exactly what you just did.
    #
    # ---
    #
    # You’re doing the *right* thing by asking this question.
    # This chapter is teaching the **intuition** behind Bayesian inference, not the full machinery.
    #
    # And that intuition — “simulate from the prior, simulate data, condition on what matches reality”.  Is the
    # beating heart of Bayesian thinking.
    #
    # So yes, this is Bayesian data analysis.
    # It’s just the **gentle on‑ramp**.
    #
    # """

    # define parameters
    n_samples = 100000
    n_ads_shown = 100

    # define generative model
    proportion_clicks = np.random.uniform(low=0.0, high=0.2, size=n_samples)
    n_visitors = np.random.binomial(n=n_ads_shown , p = proportion_clicks, size=n_samples)

    # sample data characteristics
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

    print(prior.head(10))

    sns.jointplot(data=prior,
                  x=n_visitors,
                  y=proportion_clicks,
                    )
    plt.ylabel('Proportion of Clicks')
    plt.xlabel('Number of visitors')
    plt.title('Marginal Plot of Proportion of Visitors vs Number of Visitors')
    plt.tight_layout()
    plt.show()


    # Now assune that number of visitors, n_visitor, is 13.
    n_visitors = 13
    posterior = prior[prior['n_visitors'] == n_visitors]
    sns.histplot(posterior['proportion_clicks'])
    plt.xlabel('Proportion of Clicks')
    plt.ylabel('Number')
    plt.title('Histogram of Proportion of Clicks for n_visitors = 13')
    plt.tight_layout()
    plt.show()


   # *****************************************************************************
   # demonstrate correct simulation of the posterior with a uniform prior,
    # using grid approximation:


    # for a uniform Prior
    prior_uniform = stats.uniform(loc=0.0, scale=0.2)

    plot_grid_posterior(
        prior_dist=prior_uniform,
        likelihood_fn=binomial_likelihood(n=100, k=13),
        xlim=(0, 0.25),
        title="Uniform Prior × Binomial Likelihood → Posterior",
        prior_label="Uniform Prior"
    )


    # for a beta Prior
    prior_beta = stats.beta(5, 95)

    plot_grid_posterior(
        prior_dist=prior_beta,
        likelihood_fn=binomial_likelihood(n=100, k=13),
        xlim=(0, 0.25),
        title="Beta Prior × Binomial Likelihood → Posterior",
        prior_label="Beta Prior"
    )


    # Use an approximate method to find the Expectation value of the posterior
    # Don't use grids, but do use sampling of the distributions.
    # Also, use the following:
    # Take the logarithm of the likelihood to avoid numerical underflow or overflow
    # Subtracting the maximum log-likelihood to shift all log weights so the largest one becomes 0.
    # Then exponentiating gives values between 0 and 1, instead of astronomically large or tiny numbers.
    # Normalization (dividing by the sum) restores the correct relative weights.
    # This trick is standard in numerical probability work — often called the log‑sum‑exp trick.
    # Now get the expectation value of the posterior.


    num_samples = 100000
    # n_ads_shown = 100
    # Data and prior
    a, b = 5, 95
    n, k = 100, 13

    # Sample from the prior
    p_samples = prior_beta.rvs(size=num_samples, random_state=rng)

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


    # --------------------------------------------------------------------------
    # Monte Carlo Simulation: Generating the Posterior Distribution
    # --------------------------------------------------------------------------
    # To visualize the full posterior distribution (instead of just the mean),
    # we can use the weights to resample from our prior samples.
    # This is effectively "Importance Sampling" where the Prior is the proposal.

    # Resample from p_samples based on the calculated weights
    # This creates a new set of samples that follows the Posterior distribution

    num_samples2 = 100000

    posterior_samples = rng.choice(p_samples, size=num_samples2, replace=True, p=weights_normalized)

    # Plot the results
    plt.figure(figsize=(8, 5))

    # Histogram of the Monte Carlo samples
    sns.histplot(posterior_samples,
                 stat="density",
                 bins=50,
                 kde=True,
                 label="Monte Carlo Posterior",
                 color="skyblue",
                 alpha=0.6)

    # Overlay the exact analytic solution for comparison
    x_vals = np.linspace(0, 1, 500)
    exact_pdf = stats.beta(a + k, b + n - k).pdf(x_vals)
    ax = sns.lineplot(x=x_vals,
                 y=exact_pdf,
                 color='red',
                 label="Exact Posterior",
                 linestyle="--")

    ax.set_xlim(0,0.25)
    plt.legend()
    plt.xlabel("Probability of Click, p")
    plt.ylabel("Density")
    plt.title("Monte Carlo Approximation vs. Exact Posterior")
    plt.show()
    del ax


    # Working the same problem with PyMC with Beta Prior
    # --------------------------------------------------------------------------
    # PyMC Solution (Probabilistic Programming)
    # --------------------------------------------------------------------------
    # Instead of manually calculating weights or grids, we define the model
    # and let PyMC's sampler (MCMC) find the posterior distribution for us.

    a, b = 5, 95
    n, k = 100, 13

    with pm.Model():
        # 1. Prior: Define the Beta prior for 'p'
        p_param = pm.Beta("p", alpha=a, beta=b)

        # 2. Likelihood: Define the Binomial likelihood given observed data
        #    observed=k binds this variable to our actual data.
        pm.Binomial("obs", n=n, p=p_param, observed=k)

        # 3. Inference: Draw samples from the posterior
        #    PyMC automatically chooses the NUTS sampler for continuous variables.
        idata = pm.sample(draws=2000, chains=4, random_seed=42)

    # 4. Visualization

    # first I extract the max density
    # Extract posterior samples
    samples = idata["posterior"]["p"].values.flatten()

    # Compute KDE
    kde = stats.gaussian_kde(samples)
    xs = np.linspace(0, 1, 500)
    ys = kde(xs)
    # compute max density
    max_x = xs[np.argmax(ys)]
    max_y = np.max(ys)

    #    Plot the posterior distribution and the HDI (Highest Density Interval)
    ax = az.plot_posterior(idata,
                      var_names=["p"],
                      hdi_prob=0.98,
                      group="posterior",
                      ref_val=(a + k) / (a + b + n),
                      )


    # Show the y-axis and set a label

    ax.get_yaxis().set_visible(True)
    ax.set_ylabel("Probability Density")
    ax.tick_params(axis='y', left=True, labelleft=True)  # Ensure ticks and labels are on

    ax.annotate(f"max density ≈ {max_y:.2f}",
                xy=(max_x, max_y),
                xytext=(max_x + 0.05, max_y + 0.05),
                arrowprops=dict(arrowstyle="->"))

    plt.title("Posterior Estimation using PyMC")
    plt.show()
    del ax

    # Print a statistical summary
    print("")
    print("Summary of PyMC Model:")
    print(az.summary(idata, var_names=["p"]))
    del idata


    # Working the same problem with PyMC, but with Uniform Prior
    # --------------------------------------------------------------------------
    # PyMC Solution (Probabilistic Programming)
    # --------------------------------------------------------------------------
    # Instead of manually calculating weights or grids, we define the model
    # and let PyMC's sampler (MCMC) find the posterior distribution for us.



    lower, upper = 0, 0.2
    n, k = 100, 13

    with pm.Model():
        # 1. Prior: Define the Uniform prior for 'p'
        p_param = pm.Uniform("p", lower, upper)

        # 2. Likelihood: Define the Binomial likelihood given observed data
        #    observed=k binds this variable to our actual data.
        pm.Binomial("obs", n=n, p=p_param, observed=k)

        # 3. Inference: Draw samples from the posterior
        #    PyMC automatically chooses the NUTS sampler for continuous variables.
        idata = pm.sample(draws=2000, chains=4, random_seed=42)

    # 4. Visualization

    # first I extract the max density
    # Extract posterior samples
    samples = idata["posterior"]["p"].values.flatten()

    # Compute KDE
    kde = stats.gaussian_kde(samples)
    xs = np.linspace(0, 1, 500)
    ys = kde(xs)
    # compute max density
    max_x = xs[np.argmax(ys)]
    max_y = np.max(ys)

    #    Plot the posterior distribution and the HDI (Highest Density Interval)
    ref_val = idata["posterior"]["p"].mean().item()
    ax = az.plot_posterior(idata,
                      var_names=["p"],
                      hdi_prob=0.98,
                      group="posterior",
                      ref_val=ref_val,
                      )


    # Show the y-axis and set a label

    ax.get_yaxis().set_visible(True)
    ax.set_ylabel("Probability Density")
    ax.tick_params(axis='y', left=True, labelleft=True)  # Ensure ticks and labels are on

    ax.annotate(f"max density ≈ {max_y:.2f}",
                xy=(max_x, max_y),
                xytext=(max_x + 0.05, max_y + 0.05),
                arrowprops=dict(arrowstyle="->"))

    plt.title("Posterior Estimation using PyMC")
    plt.show()
    del ax

    # Print a statistical summary
    print("")
    print("Summary of PyMC Model:")
    print(az.summary(idata, var_names=["p"]))
    del idata


if __name__ == "__main__":

    # Verify PyTensor Configuration
    print(f"--- PyTensor Mode: {pytensor.config.mode} ---")
    if pytensor.config.mode == "NUMBA":
        import warnings

        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"pytensor\.link\.c\.cmodule"
        )

    try:
        import numba
        print(f"--- Numba Version: {numba.__version__} ---")
    except ImportError:
        print("--- [!] Numba not found. PyTensor may fall back to default mode. ---")
    print()

    # define parameters
    # add a comment here for freeze_support
    freeze_support()
    main()
