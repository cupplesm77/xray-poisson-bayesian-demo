# functions.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma
import arviz as az
from pathlib import Path

def load_csv_with_schema(data_path, expected_columns, dtype_map):
    """
    Load and validate a CSV file against a specific schema.

    This utility ensures data integrity by checking for file existence,
    stripping whitespace from headers, enforcing data types, and
    verifying that the column structure matches the expected format.
    It also checks for missing values to prevent downstream errors in
    the Bayesian inference pipeline.

    Parameters
    ----------
    data_path : str or Path
        Relative or absolute path to the CSV file.
    expected_columns : list of str
        List of column names that MUST be present in the file.
    dtype_map : dict
        A dictionary mapping column names to their expected types
        (e.g., {"observation": "int"}).

    Returns
    -------
    toy_data : pd.DataFrame
        The cleaned and validated DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the specified path.
    RuntimeError
        If pandas fails to read the file (e.g., encoding issues).
    ValueError
        If the columns don't match the expected_columns or if
        missing values (NaNs) are detected.
    """

    # Pathlib Integration: Uses pathlib.Path for robust, cross-platform file path handling.
    csv_path = Path(data_path)

    # 1. Check if the file exists
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # 2. Define expected schema
    expected_cols = expected_columns
    dtype_map = dtype_map

    # 3. Read with protection
    try:
        toy_data = pd.read_csv(
            csv_path,
            encoding="utf-8",
            header=0,
            dtype=dtype_map
        )
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")

    # 4. Validate columns
    # perform fundamental data cleaning as part of the validation
    toy_data.columns = toy_data.columns.str.strip()
    # check for expected column headers
    if list(toy_data.columns) != expected_cols:
        raise ValueError(f"Unexpected columns: {toy_data.columns}")

    # 5. Validate missing values
    if toy_data.isnull().any().any():
        raise ValueError("CSV contains missing values — inspect before proceeding.")

    return toy_data


def simulate_posterior_poisson(
        n_draws: int = 100_000,
        observed_count: int = 10,
        prior_distribution: str = "uniform",
        prior_min: float = 0.0,
        prior_max: float = 80.0,
        loc: float = 0.0,
        scale: float = 1.0,
        previous_posterior: np.ndarray | None = None,
        seed: int | np.random.Generator | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a posterior distribution for a Poisson rate parameter λ
    using generative conditioning (likelihood-based rejection sampling).

    Parameters
    ----------
    n_draws : int, optional
        Number of prior samples to draw.
    observed_count : int
        Observed Poisson count.
    prior_distribution: {"uniform", "normal", "posterior"}
        Choice of prior distribution for λ.
        - "uniform": Uniform(prior_min, prior_max)
        - "normal": Normal(loc, scale), truncated to λ > 0
        - "posterior": use previous_posterior as the prior
    prior_min : float, optional
        Lower bound for Uniform prior.
    prior_max : float, optional
        Upper bound for Uniform prior.
    loc : float, optional
        Mean of Normal prior.
    scale : float, optional
        Standard deviation of Normal prior.
    previous_posterior : array-like or None
        Posterior samples from a previous update (required if prior_distribution="posterior").
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    prior_samples : np.ndarray
        Samples drawn from the chosen prior distribution.
    posterior_samples : np.ndarray
        Subset of prior samples whose simulated Poisson counts
        match the observed count.
    """

    # Robust Generator initialization
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------------
    # PRIOR SAMPLING
    # ----------------------------------------------------------------------
    if prior_distribution == "uniform":
        prior_samples = rng.uniform(prior_min, prior_max, size=n_draws)

    elif prior_distribution == "normal":
        prior_samples = rng.normal(loc=loc, scale=scale, size=n_draws)
        prior_samples = prior_samples[prior_samples > 0]

        # ensure enough positive samples
        n_extra = 0
        while len(prior_samples) < n_draws:
            n_extra += 1
            extra = rng.normal(loc=loc, scale=scale, size=n_draws)
            extra = extra[extra > 0]
            prior_samples = np.concatenate([prior_samples, extra])
        print(f"Number of extra samples for Normal prior: {n_extra}")
        prior_samples = prior_samples[:n_draws]

    elif prior_distribution == "posterior":
        if previous_posterior is None:
            raise ValueError(
                "prior_distribution='posterior' requires previous_posterior to be provided."
            )

        previous_posterior = np.asarray(previous_posterior)

        # resample if needed
        if len(previous_posterior) < n_draws:
            prior_samples = rng.choice(previous_posterior, size=n_draws, replace=True)
        else:
            prior_samples = previous_posterior

    else:
        raise ValueError(
            f"Unsupported prior distribution '{prior_distribution}'. "
            "Choose 'uniform', 'normal', or 'posterior'."
        )

    # ----------------------------------------------------------------------
    # LIKELIHOOD SIMULATION
    # ----------------------------------------------------------------------
    simulated_counts = rng.poisson(lam=prior_samples)

    # ----------------------------------------------------------------------
    # POSTERIOR EXTRACTION
    # ----------------------------------------------------------------------
    posterior_samples = prior_samples[simulated_counts == observed_count]

    if len(posterior_samples) == 0:
        import warnings
        warnings.warn(
            f"Zero samples accepted for observed_count={observed_count}. "
            "The prior may not overlap with the likelihood (Sample Collapse)."
        )

    return prior_samples, posterior_samples

# Plot a histogram
def plot_histograms(prior_samples,
                    posterior_samples,
                    observed_count=None,
                    bins=50,
                    title=None,
                    save_path=None,
                    ):
    """
    Plot prior and posterior histograms for visual comparison.
    """

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].hist(prior_samples, bins=bins, color="gray", alpha=0.7)
    ax[0].set_title("Prior Distribution for λ")
    ax[0].set_xlabel("λ (mean clicks per day)")
    ax[0].set_ylabel("Frequency")

    if title is None and observed_count is not None:
        title = f"Posterior Distribution for λ | y = {observed_count}"

    ax[1].hist(posterior_samples, bins=bins, color="steelblue", alpha=0.8)
    ax[1].set_title(title)
    ax[1].set_xlabel("λ (mean clicks per day)")
    ax[1].set_ylabel("Frequency")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to: {save_path}")

    plt.show()


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
    save_path=None,
):
    """
    Plot a posterior density using preLiz

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
        Title for the plot.
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
            # Handle cases where multiple axes might be returned (e.g., if idata is InferenceData)
            plt.suptitle(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Density plot saved to: {save_path}")

    plt.show()


# Sequential Bayesian Updating Driver
# This driver:
# - Accepts a list of observed counts
# - Automatically chooses the correct prior for each step
# - Stores priors, posteriors, and diagnostics
# - Works seamlessly with your unified simulate_posterior_poisson function
# - Produces a clean record of the entire inference process

def sequential_update_poisson(
    observations: list[int],
    n_draws=100_000,
    prior_distribution="uniform",
    prior_min=0.0,
    prior_max=80.0,
    loc=0.0,
    scale=1.0,
    seed: int | np.random.Generator | None = 42,
) -> list[dict]:
    """
    Perform sequential Bayesian updating for a Poisson rate parameter λ
    using generative conditioning.

    Parameters
    ----------
    observations : list or array-like
        Sequence of observed Poisson counts.
    n_draws : int
        Number of samples for each update.
    prior_distribution : {"uniform", "normal"}
        Initial prior type for the first update.
    prior_min, prior_max : float
        Bounds for Uniform prior.
    loc, scale : float
        Parameters for Normal prior.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    history : list of dict
        Each element contains:
        - "step": update index
        - "observed": observed count
        - "prior_samples": samples used as prior
        - "posterior_samples": resulting posterior
    """

    rng = np.random.default_rng(seed)
    history = []

    previous_posterior = None

    for step, obs in enumerate(observations, start=1):

        # Determine which prior to use
        if step == 1:
            pdist = prior_distribution
        else:
            pdist = "posterior"

        prior_samples, posterior_samples = simulate_posterior_poisson(
            n_draws=n_draws,
            observed_count=obs,
            prior_distribution=pdist,
            prior_min=prior_min,
            prior_max=prior_max,
            loc=loc,
            scale=scale,
            previous_posterior=previous_posterior,
            seed=rng.integers(0, 1_000_000_000),
        )

        history.append(
            {
                "step": step,
                "observed": obs,
                "prior_samples": prior_samples,
                "posterior_samples": posterior_samples,
            }
        )

        previous_posterior = posterior_samples

    return history

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


def summarize_posterior(posterior_samples: np.ndarray,
                        ci: float = 0.95,
                        ) -> dict:
    """
    Compute summary statistics for posterior samples of λ.

    Parameters
    ----------
    posterior_samples : np.ndarray
        Array of samples from the posterior distribution.
    ci : float
        Credible interval width (e.g., 0.95 for 95%).

    Returns
    -------
    Dict containing mean, median, and credible interval tuple.
    """
    if len(posterior_samples) == 0:
        return {"mean": np.nan, "median": np.nan, "ci": (np.nan, np.nan)}

    lower_q = (1 - ci) / 2
    upper_q = 1 - lower_q

    return {
        "mean": np.mean(posterior_samples),
        "median": np.median(posterior_samples),
        "ci": (np.quantile(posterior_samples, lower_q),
               np.quantile(posterior_samples, upper_q)),
    }