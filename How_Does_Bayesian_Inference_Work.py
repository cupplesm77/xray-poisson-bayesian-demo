# How_Does_Bayesian_Inference_Work.py
from multiprocessing.spawn import freeze_support

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import MatplotlibDeprecationWarning
# from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import seaborn as sns
import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

rng = np.random.default_rng(42)


def main():

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


    # demonstrate correct simulation of the posterior with a uniform prior:

    # Data and prior
    n, k = 100, 13

    # 1) Prior distribution object
    prior_uniform = stats.uniform(loc=0.0, scale=0.2)

    # 2) Candidate grid of p values
    p_grid = np.linspace(0, 1, 500)

    # 3) Prior density at each grid point
    prior_density = prior_uniform.pdf(p_grid)

    # 4) Likelihood at each grid point (Binomial, dropping constant C(n,k))
    likelihood = (p_grid**k) * ((1 - p_grid)**(n - k))

    # 5) Unnormalized posterior = prior_density × likelihood
    unnormed_posterior = prior_density * likelihood

    # 6) Normalize to make it a proper distribution
    posterior = unnormed_posterior / np.trapezoid(unnormed_posterior, p_grid)

    # Plot
    fig, ax = plt.subplots(figsize=(8,5))
    sns.lineplot(x=p_grid,
                 y=unnormed_posterior,
                 label="Unnormalized posterior",
                 ax=ax)
    sns.lineplot(x=p_grid,
                 y=posterior,
                 label="Normalized posterior",
                 ax=ax)
    sns.lineplot(x=p_grid,
                 y=prior_density,
                 label=" Uniform Prior",
                 linestyle="--",
                 ax=ax)
    plt.legend()
    plt.xlabel("probability of click, p")
    plt.ylabel("Probability Density")
    plt.title("Uniform Prior × Likelihood → Uniform Posterior")

    ax.set_xlim(0, 0.25)
    plt.show()
    del ax


    # demonstrate correct simulation of the posterior with a Beta prior:

    # Data and prior
    a, b = 5, 95
    n, k = 100, 13

    # 1) Prior distribution object
    prior_beta = stats.beta(a, b)

    # 2) Candidate grid of p values
    p_grid = np.linspace(0, 1, 500)

    # 3) Prior density at each grid point
    prior_density = prior_beta.pdf(p_grid)

    # 4) Likelihood at each grid point (Binomial, dropping constant C(n,k))
    likelihood = (p_grid**k) * ((1 - p_grid)**(n - k))

    # 5) Unnormalized posterior = prior_density × likelihood
    unnormed_posterior = prior_density * likelihood

    # 6) Normalize to make it a proper distribution
    posterior = unnormed_posterior / np.trapezoid(unnormed_posterior, p_grid)

    # Plot
    fig, ax = plt.subplots(figsize=(8,5))
    sns.lineplot(x=p_grid,
                 y=unnormed_posterior,
                 label="Unnormalized posterior",
                 ax=ax)
    sns.lineplot(x=p_grid,
                 y=posterior,
                 label="Normalized posterior",
                 ax=ax)
    sns.lineplot(x=p_grid,
                 y=prior_density,
                 label=" Beta Prior",
                 linestyle="--",
                 ax=ax)
    plt.legend()
    plt.xlabel("probability of click, p")
    plt.ylabel("Probability Density")
    plt.title("Beta Prior × Likelihood →  Beta Posterior")

    ax.set_xlim(0, 0.25)
    plt.show()
    del ax


    # Now demonstrate the exact solution using a beta-binomial model:
    num_samples = 100_000
    a, b = 5, 95
    posterior = stats.beta(a + k, b + n - k)
    # Prior density at each grid point
    prior_beta = stats.beta(a, b)
    prior__beta_density = prior_beta.pdf(p_grid)
    plt.figure(figsize=(8,5))
    ax1 = sns.lineplot(x=p_grid, y=posterior.pdf(p_grid), label="Exact posterior")
    ax2 = sns.lineplot(x=p_grid, y=prior__beta_density, label="Prior", linestyle="--")
    plt.legend()
    plt.xlabel("probability of click, p")
    plt.ylabel("Probability Density")
    plt.title("Exact Posterior vs Prior")

    ax1.set_xlim(0, 0.25)
    ax2.set_xlim(0, 0.25)
    plt.show()
    del ax1, ax2

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

    # Working the same problem with PyMC
    # --------------------------------------------------------------------------
    # PyMC Solution (Probabilistic Programming)
    # --------------------------------------------------------------------------
    # Instead of manually calculating weights or grids, we define the model
    # and let PyMC's sampler (MCMC) find the posterior distribution for us.

    try:
        import pymc as pm
        import arviz as az  # Companion library for visualizing PyMC results

        print("\n--- Running PyMC Model ---")

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

    except ImportError:
        print("\n[!] PyMC or Arviz library not found.")
        print("    To run the PyMC example, please install them: pip install pymc arviz")


    # Can you plot the posterior distribution in the same way that you plotted it in the previous example?
    # How is this data different from the previous computations for the posterior?

if __name__ == "__main__":
    # add a comment here for freeze_support

    freeze_support()
    main()
