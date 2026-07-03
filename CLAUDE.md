# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jupyter Book repository containing educational notebooks demonstrating nested sampling algorithms from the BlackJAX library. The book focuses on physics-motivated use cases and provides pedagogical examples of nested sampling implementations.

The nested sampling code now lives in **main BlackJAX** (https://github.com/blackjax-devs/blackjax), in the `blackjax.ns` subpackage, with top-level entry points `blackjax.nss` (Nested Slice Sampling) and `blackjax.nsswig` (slice-within-Gibbs). It is **not yet in a tagged PyPI release**, so it must be installed from the `main` branch with `git` (see Installation). The older `handley-lab/blackjax` fork (`v0.1.0-beta`) is deprecated — do not reference it in new content.

BlackJAX maintains a complementary [Sampling Book](https://blackjax-devs.github.io/sampling-book/) whose [Nested Sampling chapter](https://blackjax-devs.github.io/sampling-book/algorithms/nested-sampling/) is the upstream companion to this book; keep the API usage here consistent with it.

## Common Development Commands

### Building the Book
```bash
# Install dependencies
pip install -r requirements.txt

# Build the book locally
jupyter-book build .
# or shorthand
jb build .

# The built HTML will be in _build/html/
```

### Installation
```bash
# Install nested sampling from main BlackJAX (not yet on PyPI, so git is required)
pip install git+https://github.com/blackjax-devs/blackjax.git

# Install visualization dependencies (for examples)
pip install anesthetic
```

### Contributing a New Example

1. Create a notebook in the appropriate directory:
   - `basic/` - Simple introductory examples
   - `advanced/` - More involved implementations (posterior repartitioning, Random Walk NS from primitives)
   - `physics/` - Physics-specific applications (supernovae, cosmology)
   - `scripts/` - Python scripts for standalone examples

2. Add your notebook to `_toc.yml` under the appropriate section

3. Add yourself to `contributors.md`

4. Include citations in `references.bib` and use `{cite}`citation_key`` in markdown

5. Test the build locally with `jb build .`

## Repository Structure

- **Notebooks**: Interactive examples demonstrating nested sampling usage
  - Examples are pre-executed (notebooks run statically due to `execute_notebooks: 'off'`)
  - Visual state is preserved in notebooks for display in the book
  
- **Scripts**: Standalone Python implementations for specific physics problems
  - `supernovae.py` - SALT model fitting for supernovae
  - `CMB.py`, `GW.py` - Cosmological and gravitational wave examples
  - `BF.py` - Bayes factor computation

- **Configuration**:
  - `_config.yml` - Jupyter Book configuration
  - `_toc.yml` - Table of contents structure
  - GitHub Actions workflow in `.github/workflows/build_deploy.yaml` for automatic deployment

## Key Patterns

### Nested Sampling with BlackJAX

Standard workflow for nested sampling:
```python
import blackjax
from blackjax.ns.utils import finalise

# Define likelihood and prior
loglikelihood_fn = lambda x: ...
logprior_fn = lambda x: ...

# Initialize algorithm
algo = blackjax.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    num_delete=50,
    num_inner_steps=20,
)

# Run sampling loop
state = algo.init(initial_particles)
dead_points = []
while not state.integrator.logZ_live - state.integrator.logZ < -3:
    rng_key, subkey = jax.random.split(rng_key)
    state, info = algo.step(subkey, state)   # step returns (new_state, NSInfo)
    dead_points.append(info)

# Finalize results: returns an NSInfo whose `.particles` is a StateWithLogLikelihood
ns_run = finalise(state, dead_points)
```

Key API notes for main BlackJAX (these changed from the old fork):
- Running evidence lives on `state.integrator`: use `state.integrator.logZ` and `state.integrator.logZ_live` (not `state.logZ` / `state.logZ_live`). The standard stopping rule is `logZ_live - logZ < -3`.
- `finalise(...)` returns an `NSInfo`; the particles are under `.particles`, so use `ns_run.particles.position`, `ns_run.particles.loglikelihood`, `ns_run.particles.loglikelihood_birth`.
- `blackjax.ns.utils.sample(...)` returns a particle state — take `.position` for the resampled positions. `log_weights`/`ess`/`uniform_prior` also live in `blackjax.ns.utils`.
- Custom inner kernels: use `blackjax.ns.from_mcmc.reject_constrained_step` + `blackjax.ns.from_mcmc.build_kernel` (the old `repeat_kernel` / `new_state_and_info` helpers no longer exist).

### Visualization with Anesthetic

Convert results to anesthetic format for plotting:
```python
import anesthetic

nested_samples = anesthetic.NestedSamples(
    data=ns_run.particles.position,
    logL=ns_run.particles.loglikelihood,
    logL_birth=ns_run.particles.loglikelihood_birth,
)
```

## Dependencies

Core dependencies (from `requirements.txt`):
- `jupyter-book` - For building the book
- `matplotlib` - For plotting
- `numpy` - For numerical operations

Example notebooks may require additional packages:
- `blackjax` (from main, via git — see Installation)
- `anesthetic` - For nested sampling visualization
- `jax`, `jax.numpy` - JAX framework
- Physics-specific packages (e.g., `cosmopower-jax` for the CMB example, `jax-bandflux`/`jax_supernovae` for supernovae)

## Notes

- The book is deployed automatically to GitHub Pages via GitHub Actions on push to main branch
- Notebooks are NOT re-executed during build (static display mode) to avoid dependency management issues
- Examples focus on JAX-based implementations for hardware acceleration
- Citations follow standard academic format using BibTeX references