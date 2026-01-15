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

from pathlib import Path

from matplotlib import pyplot as plt

from functions import (
    simulate_posterior_poisson,
    plot_histograms,
    analytic_posterior_gamma,
    plot_posterior_density,
    sequential_update_poisson,
    load_csv_with_schema,
    summarize_posterior,
)

if __name__ == "__main__":

    # Load the toy-data dataframe

    plot_path = "plots"
    toy_data_path = "data/toy_data.csv"
    expected_cols = ["observation","counts_per_exposure"]
    dtype_map = {"observation": "int", "counts_per_exposure": "int"}
    toy_data = load_csv_with_schema(toy_data_path,
                                    expected_cols,
                                    dtype_map,
                                    )

    print(toy_data.head())
    print("")

    # Sequential Baysesian Driver
    observations = toy_data["counts_per_exposure"].to_list()
    print(observations)

    # Create a container for simulation history
    history = sequential_update_poisson(
        observations,
        n_draws=100_000,
        prior_distribution="uniform",
        prior_min=0,
        prior_max=80,
    )

    # Prepare directory for plots
    plot_dir = Path("plots/")
    plot_dir.mkdir(exist_ok=True)  # Ensure directory exists

    # Driver loop
    for i, his in enumerate(history, start=0):
        n_bins = 30
        # capture the observation number
        num_observation = his['observed']
        step = his['step']

        # Define save paths using Path for cross-platform compatibility
        histogram_path = plot_dir / f"step_{step}_obs_{num_observation}_histogram.png"
        density_path = plot_dir / f"step_{step}_obs_{num_observation}_density.png"


    # histograms of prior and posterior samples
        plot_histograms(his["prior_samples"],
                        his["posterior_samples"],
                        observed_count=num_observation,
                        bins=n_bins,
                        title=f"Posterior Distribution for λ | y = {num_observation}",
                        save_path=histogram_path,
                        )

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
                save_path=density_path,
        )
        plt.close('all') # Better than 'del' for freeing Matplotlib memory


    # Now for the Inference Part of the Routine.

    # ------------------------------------------------------------
    # Inference Summary from the Final Posterior
    # ------------------------------------------------------------
    final_posterior = history[-1]["posterior_samples"]
    summary = summarize_posterior(final_posterior, ci=0.95)

    print("\n=== Bayesian Inference Summary (Toy Example) ===")
    print(f"Observed counts: {observations}")
    print(f"Posterior mean of λ:    {summary['mean']:.2f}")
    print(f"Posterior median of λ:  {summary['median']:.2f}")
    print(f"95% credible interval:  [{summary['ci'][0]:.2f}, {summary['ci'][1]:.2f}]")


    print("\nInterpretation:")
    print(
        "Based on the three observed exposures [3, 5, 2], the posterior distribution \n"
        "suggests that the underlying photon rate λ for this faint X‑ray source is \n"
        f"most plausibly in the range {summary['ci'][0]:.1f} to {summary['ci'][1]:.1f} counts per exposure. \n"
        "The posterior mean and median both fall near the center of this interval, indicating \n"
        "that the sequential Bayesian update has stabilized toward a consistent estimate of the \n"
        "source brightness. Although the data are sparse, the posterior distribution provides a \n"
        "coherent quantification of uncertainty that would naturally tighten as additional exposures \n"
        "are incorporated.\n"
    )