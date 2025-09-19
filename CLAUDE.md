# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jupyter Book repository containing educational notebooks demonstrating nested sampling algorithms from the BlackJAX library. The book focuses on physics-motivated use cases and provides pedagogical examples of nested sampling implementations.

The repository uses the nested sampling fork of BlackJAX from: https://github.com/handley-lab/blackjax (branch: `nested_sampling`)

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
# Install the nested sampling fork of BlackJAX
pip install git+https://github.com/handley-lab/blackjax@nested_sampling

# Install visualization dependencies (for examples)
pip install anesthetic
```

### Contributing a New Example

1. Create a notebook in the appropriate directory:
   - `basic/` - Simple introductory examples
   - `advanced/` - Complex implementations (GP, Random Walk NS)
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
while not converged:
    state, dead_point = algo.step(rng_key, state)
    # collect dead points

# Finalize results
final_state = finalise(state, dead_points)
```

### Visualization with Anesthetic

Convert results to anesthetic format for plotting:
```python
import anesthetic

nested_samples = anesthetic.NestedSamples(
    data=final_state.particles,
    logL=final_state.loglikelihood,
    logL_birth=final_state.loglikelihood_birth,
)
```

## Dependencies

Core dependencies (from `requirements.txt`):
- `jupyter-book` - For building the book
- `matplotlib` - For plotting
- `numpy` - For numerical operations

Example notebooks may require additional packages:
- `blackjax` (nested sampling fork)
- `anesthetic` - For nested sampling visualization
- `jax`, `jax.numpy` - JAX framework
- Physics-specific packages (e.g., `jax_supernovae` for supernovae examples)

## Notes

- The book is deployed automatically to GitHub Pages via GitHub Actions on push to main branch
- Notebooks are NOT re-executed during build (static display mode) to avoid dependency management issues
- Examples focus on JAX-based implementations for hardware acceleration
- Citations follow standard academic format using BibTeX references