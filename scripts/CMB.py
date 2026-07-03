# | # Minimal example of CMB parameter estimation
# |
# | This script provides a minimal working example of using nested sampling (via BlackJAX) for CMB parameter estimation. It does the following:
# |
# | • Installs and imports the required libraries (jax, blackjax, anesthetic, cosmopower_jax, etc.).
# | • Defines a CMB class that models CMB data (including a method to generate random realizations and a log-likelihood function based on a χ² distribution across multipoles).
# | • Uses a CosmoPowerJAX emulator trained for CMB temperature power spectra.
# | • Sets up a uniform prior over six cosmological parameters and a likelihood function that compares the emulator’s prediction to observed CMB data.
# | • Initializes and runs the BlackJAX nested sampling algorithm with an adaptive procedure, stepping through "live" points until convergence.
# | • Collects "dead" and "live" points, then uses Anesthetic’s NestedSamples to post-process and write the samples (with their log-likelihoods) to a CSV file.
# |
# |
# | ## Installation
# |```bash
# | python -m venv venv
# | source venv/bin/activate
# | pip install tqdm numpy jax anesthetic cosmopower_jax
# | pip install git+https://github.com/blackjax-devs/blackjax.git
# | python CMB.py
# |```
# | The code takes about 90s to run on an L4 GPU (~250 dead points/second).

import blackjax
import blackjax.ns.adaptive
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from anesthetic import NestedSamples
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX

jax.config.update("jax_enable_x64", True)

# | Define CMB data


class CMB(object):
    def __init__(self, Cl):
        self.Cl = Cl
        self.l = jnp.arange(2, 2509)

    def rvs(self, shape=()):
        shape = tuple(jnp.atleast_1d(shape))
        return (
            jax.random.chisquare(
                jax.random.PRNGKey(0), 2 * self.l + 1, shape + self.Cl.shape
            )
            * self.Cl
            / (2 * self.l + 1)
        )

    def logpdf(self, x):
        return (
            jax.scipy.stats.chi2.logpdf(
                (2 * self.l + 1) * x / self.Cl, 2 * self.l + 1
            )
            + jnp.log(2 * self.l + 1)
            - jnp.log(self.Cl)
        ).sum(axis=-1)


np.random.seed(0)
emulator = CosmoPowerJAX(probe="cmb_tt")
planckparams = jnp.array([0.02225, 0.1198, 0.693, 0.097, 0.965, 3.05])
d_obs = CMB(emulator.predict(planckparams)).rvs()


columns = ["Ωbh2", "Ωch2", "h", "τ", "ns", "lnA"]
labels = [
    r"$\Omega_b h^2$",
    r"$\Omega_c h^2$",
    r"$h$",
    r"$\tau$",
    r"$n_s$",
    r"$\ln(10^{10}A_s)$",
]

# | Define the prior function
parammin, parammax = jnp.array(
    [
        [0.01865, 0.02625],
        [0.05, 0.255],
        [0.64, 0.82],
        [0.04, 0.12],
        [0.84, 1.1],
        [1.61, 3.91],
    ]
).T  ## Prior set by the emulator's training range (cosmo_power_jax)


def logprior_fn(x):  ## 6D Uniform prior
    return jax.scipy.stats.uniform.logpdf(
        x, parammin, (parammax - parammin)
    ).sum()


# | Define the likelihood function
@jax.jit
def loglikelihood_fn(x):
    return CMB(jnp.array(emulator.predict(x))).logpdf(d_obs)


# | Define the Nested Sampling algorithm
n_dims = len(columns)
n_live = 1000
n_delete = 500
num_mcmc_steps = n_dims * 3

# | Initialize the Nested Sampling algorithm
nested_sampler = blackjax.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    num_delete=n_delete,
    num_inner_steps=num_mcmc_steps,
)


@jax.jit
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = nested_sampler.step(subk, state)
    return (state, k), dead_point


# | Sample live points from the prior
rng_key = jax.random.PRNGKey(0)
rng_key, init_key = jax.random.split(rng_key, 2)

initial_particles = (
    jax.random.uniform(init_key, (n_live, n_dims)) * (parammax - parammin)
    + parammin
)
state = nested_sampler.init(initial_particles)

# | Run Nested Sampling
dead = []
with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    while not state.integrator.logZ_live - state.integrator.logZ < -3:
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        pbar.update(n_delete)  # Update progress bar

# | anesthetic post-processing
from blackjax.ns.utils import finalise

final = finalise(state, dead)
logL = np.asarray(final.particles.loglikelihood, dtype=float)
logL_birth = np.asarray(final.particles.loglikelihood_birth, dtype=float)
data = np.asarray(final.particles.position, dtype=float)
samples = NestedSamples(
    data, logL=logL, logL_birth=logL_birth, columns=columns, labels=labels
)
samples.to_csv("CMB.csv")
