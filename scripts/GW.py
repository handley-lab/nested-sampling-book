# | # Minimal example of GW parameter estimation
# |
# | This script performs Bayesian inference on LIGO data (from GW150914) using
# | a nested sampling algorithm implemented with BlackJAX. It loads the
# | detector data and sets up a gravitational-wave waveform model, defines a
# | prior and likelihood for the model parameters, then runs nested sampling
# | to sample from the posterior. Finally, it processes the samples with
# | anesthetic and writes them to a CSV file.
# |
# | ## Installation
# |```bash
# | python -m venv venv
# | source venv/bin/activate
# | pip install git+https://git.ligo.org/lscsoft/ligo-segments.git
# | pip install git+https://github.com/kazewong/jim
# | pip install git+https://github.com/handley-lab/blackjax@proposal
# | pip install anesthetic
# | python GW.py
# |```
# | The code takes about 12 minutes to run on an L4 GPU [~38 dead points/second].

import blackjax
import jax
import jax.numpy as jnp
import tqdm

from astropy.time import Time
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.likelihood import (
    original_likelihood as likelihood_function,
)
from jimgw.single_event.waveform import RippleIMRPhenomD

import distrax

jax.config.update("jax_enable_x64", True)

# | Define LIGO event data

gps = 1126259462.4
fmin = 20.0
fmax = 1024.0
H1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
L1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

waveform = RippleIMRPhenomD(f_ref=20)
detectors = [H1, L1]
frequencies = H1.frequencies
duration = 4
post_trigger_duration = 2
epoch = duration - post_trigger_duration
gmst = Time(gps, format="gps").sidereal_time("apparent", "greenwich").rad

column_to_label = {
    "M_c": r"$M_c$",
    "q": r"$q$",
    "s1_z": r"$s_{1z}$",
    "s2_z": r"$s_{2z}$",
    "iota": r"$\iota$",
    "d_L": r"$d_L$",
    "t_c": r"$t_c$",
    "phi": r"$\phi_c$",
    "psi": r"$\psi$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
}


@jax.jit
def loglikelihood_fn(p):
    # p = params.copy()
    p["eta"] = p["q"] / (1 + p["q"]) ** 2
    p["gmst"] = gmst
    p["phase_c"] = p["phi"]
    waveform_sky = waveform(frequencies, p)
    align_time = jnp.exp(-1j * 2 * jnp.pi * frequencies * (epoch + p["t_c"]))
    return likelihood_function(
        p, waveform_sky, detectors, frequencies, align_time
    )


# | Define the prior function

M_c_min, M_c_max = 10.0, 80.0
q_min, q_max = 0.125, 1.0
d_L_min, d_L_max = 1.0, 2000.0
t_c_min, t_c_max = -0.05, 0.05


prior = distrax.Joint(
    {
        "M_c": distrax.Uniform(low=M_c_min, high=M_c_max),
        "q": distrax.Uniform(low=q_min, high=q_max),
        "s1_z": distrax.Uniform(low=-1.0, high=1.0),
        "s2_z": distrax.Uniform(low=-1.0, high=1.0),
        "t_c": distrax.Uniform(low=t_c_min, high=t_c_max),
        "iota": distrax.Transformed(
            distrax.Beta(2, 2),
            distrax.Lambda(lambda x: x * jnp.pi),
        ),
        "dec": distrax.Transformed(
            distrax.Beta(2, 2),
            distrax.Lambda(lambda x: x * jnp.pi - jnp.pi / 2),
        ),
        "phi": distrax.Uniform(low=0.0, high=jnp.pi),
        "psi": distrax.Uniform(low=0.0, high=2.0 * jnp.pi),
        "ra": distrax.Uniform(low=0.0, high=2.0 * jnp.pi),
        "d_L": distrax.Transformed(
            distrax.Uniform(low=0.0, high=1.0),
            distrax.Lambda(
                lambda x: x ** (1.0 / 2.0) * (d_L_max - d_L_min) + d_L_min
            ),
        ),
    }
)

test_sample, ravel_fn = jax.flatten_util.ravel_pytree(
    prior.sample(seed=jax.random.PRNGKey(0))
)

# | Define the Nested Sampling algorithm
n_dims = test_sample.shape[0]
n_live = 1000
n_delete = 500
num_mcmc_steps = n_dims * 3

# | Sample live points from the prior
rng_key = jax.random.PRNGKey(0)
rng_key, init_key = jax.random.split(rng_key, 2)
particles = prior.sample(seed=init_key, sample_shape=(n_live,))

# | Initialize the Nested Sampling algorithm
nested_sampler = blackjax.ns.adaptive.nss(
    logprior_fn=prior.log_prob,
    loglikelihood_fn=loglikelihood_fn,
    n_delete=n_delete,
    num_mcmc_steps=num_mcmc_steps,
    ravel_fn=ravel_fn,
)

state = nested_sampler.init(particles, loglikelihood_fn)


@jax.jit
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = nested_sampler.step(subk, state)
    return (state, k), dead_point


# | Run Nested Sampling
dead = []
with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    while not state.sampler_state.logZ_live - state.sampler_state.logZ < -3:
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        pbar.update(n_delete)  # Update progress bar

# | anesthetic post-processing
from anesthetic import NestedSamples
import numpy as np

dead = jax.tree.map(
    lambda *args: jnp.reshape(
        jnp.stack(args, axis=0), (-1,) + args[0].shape[1:]
    ),
    *dead,
)
live = state.sampler_state

logL = np.concatenate((dead.logL, live.logL), dtype=float)
logL_birth = np.concatenate((dead.logL_birth, live.logL_birth), dtype=float)
data = np.concatenate(
    [
        np.column_stack([v for v in dead.particles.values()]),
        np.column_stack([v for v in live.particles.values()]),
    ],
    axis=0,
)

samples = NestedSamples(
    data,
    logL=logL,
    logL_birth=logL_birth,
    columns=particles.keys(),
    labels=column_to_label,
)
samples.to_csv("GW.csv")
