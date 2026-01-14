# simulate_posterior_poisson.py

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

from functions import (
    simulate_posterior_poisson,
    plot_histograms,
    analytic_posterior_gamma,
    plot_posterior_density,
    sequential_update_poisson
)

if __name__ == "__main__":

    # example case for observed counts = 12
    obs_count = 12
    # Example usage
    prior, posterior = simulate_posterior_poisson(
        n_draws=100_000,
        prior_min=0,
        prior_max=80,
        observed_count=obs_count,
        prior_distribution="uniform",
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
        observed_count=obs_count,
        prior_distribution="uniform",
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
        observed_count=obs_count,
        prior_distribution="uniform",
    )

    # Re-sample from the posterior to restore sample size
    # This prevents the sample size from collapsing to near-zero in the subsequent step
    # rng = np.random.default_rng(42)
    # resampled_prior_for_19 = rng.choice(posterior_12, size=100_000, replace=True)

    obs_count = 19
    prior_19, posterior_19 = simulate_posterior_poisson(previous_posterior=posterior_12,
                                                        observed_count=obs_count,
                                                        prior_distribution="posterior",
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
    prior_18, posterior_18 = simulate_posterior_poisson(previous_posterior=post19,
                                                        observed_count=obs_count,
                                                        prior_distribution="posterior",
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
    prior_16, posterior_16 = simulate_posterior_poisson(previous_posterior=posterior_18,
                                                        observed_count=obs_count,
                                                        prior_distribution="posterior",
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


    # Sequential Baysesian Driver Test Run
    observations = [3, 5, 2]

    history = sequential_update_poisson(
        observations,
        n_draws=100_000,
        prior_distribution="uniform",
        prior_min=0,
        prior_max=80,
    )



    for i, his in enumerate(history, start=0):
        n_bins = 30
        # capture the observation number
        num_observation = his['observed']
        # histograms of prior and posterior samples
        plot_histograms(his["prior_samples"],
                        his["posterior_samples"],
                        observed_count=num_observation,
                        bins=n_bins)

        # posterior density plot
        g = plot_posterior_density(
                his["posterior_samples"],
                "lambda",
                ci=0.98,
                color="steelblue",
                alpha=0.7,
                linewidth=2,
                facet=None,
                title=f"Posterior Density for λ Given Observed Count = {num_observation}",
        )
        del g



    # Now Build the Inference Part of the Routine.