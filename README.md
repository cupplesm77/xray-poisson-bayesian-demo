# Bayesian Sequential Updating for Poisson Count Data  
*A simulation‑based introduction using a toy X‑ray astronomy example*

## Overview
This repository demonstrates **Bayesian inference for Poisson count data** using a transparent, simulation‑based 
approach. The workflow follows the generative logic taught in introductory Bayesian courses:

1. Draw samples from a prior distribution over the Poisson rate parameter \( \lambda \)  
2. Simulate Poisson counts from each sampled \( \lambda \)  
3. Condition on the observed data by retaining only those \( \lambda \) values that could have produced the observed count  
4. Visualize and analyze the resulting posterior distribution  

This approach is implemented through **likelihood‑based rejection sampling**, a simple form of Approximate Bayesian 
Computation (ABC). The repository begins with a **toy X‑ray astronomy example** and is structured to scale naturally 
toward **real historical and current X‑ray data**.

---

## Scientific Motivation
Photon arrivals in X‑ray astronomy are well modeled as a **Poisson process**, especially in the low‑count regime 
typical of faint sources. Bayesian methods are widely used in this domain because they:

- handle low counts gracefully  
- provide calibrated uncertainty  
- support sequential updating as new exposures arrive  
- integrate naturally with physical priors and hierarchical models  

This repository implements the simplest version of that workflow to make the core ideas clear and reproducible.

---

## Toy Example: Faint X‑ray Source with Three Exposures
We begin with a controlled, pedagogical example that mirrors real scientific conditions.

### Scenario
- A faint X‑ray source is observed across **three short exposures**.  
- Photon counts are low and noisy.  
- Each exposure is modeled as:  
  \[
  y_i \sim \text{Poisson}(\lambda)
  \]
- Prior on the photon rate:  
  \[
  \lambda \sim \text{Uniform}(0, 80)
  \]

### Observed counts
```
[3, 5, 2]
```

### Goal
Perform **sequential Bayesian updating** as each exposure arrives, visualizing how the posterior distribution 
for \( \lambda \) evolves.

This toy example serves as a clean, reproducible foundation before moving to real X‑ray data.

---

## Code Structure

### Core Functions (`functions.py`)

#### 1. `simulate_posterior_poisson`
Draws prior samples, simulates Poisson counts, and extracts posterior samples.

```python
prior_samples, posterior_samples = simulate_posterior_poisson(
    n_draws=100_000,
    observed_count=5,
    prior_distribution="uniform",
    prior_min=0.0,
    prior_max=80.0,
    seed=42,
)
```

#### 2. `plot_histograms`
Plots prior and posterior histograms for visual comparison.

```python
plot_histograms(prior_samples, posterior_samples, observed_count=5, bins=30)
```

#### 3. `analytic_posterior_gamma`
Provides the analytic Gamma posterior for comparison when the prior is flat on \([0, \infty)\).

```python
mean, (lower, upper) = analytic_posterior_gamma(observed_count=5)
```

#### 4. `plot_posterior_density`
Uses ArviZ to plot posterior densities with credible intervals.

```python
idata = {"lambda": posterior_samples}
plot_posterior_density(
    idata,
    var_name="lambda",
    ci=0.94,
    color="steelblue",
)
```

#### 5. `sequential_update_poisson`
Performs sequential Bayesian updating across multiple exposures.

```python
history = sequential_update_poisson(
    observations=[3, 5, 2],
    n_draws=100_000,
    prior_distribution="uniform",
    prior_min=0.0,
    prior_max=80.0,
)
```

---

## Example Workflow (`poisson_example.py`)
The top‑level script demonstrates the full sequential updating process:

```python
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
    num_observation = his['observed']

    plot_histograms(
        his["prior_samples"],
        his["posterior_samples"],
        observed_count=num_observation,
        bins=n_bins
    )

    g = plot_posterior_density(
        his["posterior_samples"],
        "lambda",
        ci=0.98,
        color="steelblue",
        alpha=0.7,
        linewidth=2,
        title=f"Posterior Density for λ Given Observed Count = {num_observation}",
    )
    del g
```

This produces a sequence of prior/posterior histograms and posterior density plots for each exposure.

---

## Roadmap: From Toy Example to Real Data
This repository is intentionally structured to grow into a full scientific workflow.

Planned (possible) extensions include:

- Adding real photon‑count data from Chandra or XMM‑Newton  
- Performing sequential updating on real exposures  
- Comparing Bayesian posteriors with classical estimators  
- Incorporating background subtraction  
- Exploring hierarchical priors  
- Modeling time‑varying \( \lambda \) for variable sources  
- Publishing a short technical note or arXiv preprint  

The toy example serves as a clean, reproducible foundation for these future steps.

---

## References
These published sources support the scientific and statistical background:

- **van Dyk, D. A. (2025).** *Bayesian Statistical Methods for Astronomy Part I: Foundations.*  
  Discusses Poisson models, Bayesian inference, and low‑count regimes.

- **Kolaczyk, E. D.** *Bayesian Multiscale Methods for Poisson Count Data.*  
  Provides context for Poisson modeling in high‑energy astrophysics.

- **Song et al. (2025).** *A Poisson Process AutoDecoder for X‑ray Sources.*  
  Modern research confirming that photon arrivals follow a Poisson process.

---

## License
Add your preferred license here (MIT recommended for scientific code).

---

## Contributions
Contributions are welcome—especially for real‑data extensions, visualization improvements, and statistical enhancements.

---

Appendix: Why This Method Is Sometimes Called ABC
Although the Poisson likelihood is easy to evaluate analytically, the simulation‑based approach used in this 
repository follows the structure of Approximate Bayesian Computation (ABC). In ABC methods, the likelihood is 
not evaluated directly; instead, parameters are proposed from the prior, synthetic data are generated from 
the model, and parameters are accepted only when the simulated data match the observed data. This 
“simulate → compare → accept” pattern is the defining characteristic of ABC. In our toy example, we apply this 
logic even though a closed‑form posterior exists, because it provides a transparent, generative view of Bayesian 
updating and generalizes naturally to more complex models where the likelihood may be difficult or impossible 
to compute. In that sense, the method is “approximate” in its algorithmic structure, not in its philosophical 
commitment to Bayesian inference.