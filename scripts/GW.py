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
from jimgw.single_event.likelihood import original_likelihood as likelihood_function
from jimgw.single_event.waveform import RippleIMRPhenomD

jax.config.update('jax_enable_x64', True) 

# | Define LIGO event data

gps = 1126259462.4
fmin = 20.0
fmax = 1024.0
H1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
L1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

waveform = RippleIMRPhenomD(f_ref=20)
detectors = [H1, L1]
frequencies = H1.frequencies
duration=4
post_trigger_duration=2
epoch = duration - post_trigger_duration
gmst = Time(gps, format="gps").sidereal_time("apparent", "greenwich").rad

columns = ["M_c", "q", "s1_z", "s2_z", "iota", "d_L", "t_c", "phase_c", "psi", "ra", "dec"]
labels = [r"$M_c$", r"$q$", r"$s_{1z}$", r"$s_{2z}$", r"$\iota$", r"$d_L$", r"$t_c$", r"$\phi_c$", r"$\psi$", r"$\alpha$", r"$\delta$"]

@jax.jit
def loglikelihood_fn(params):
    p = params.copy()
    p["eta"] = p["q"] / (1 + p["q"])**2
    p["gmst"] = gmst
    waveform_sky = waveform(frequencies, p)
    align_time = jnp.exp(-1j * 2 * jnp.pi * frequencies * (epoch + p["t_c"]))
    return likelihood_function(p, waveform_sky, detectors, frequencies, align_time)

# | Define the prior function

M_c_min, M_c_max = 10.0, 80.0
q_min, q_max = 0.125, 1.0
d_L_min, d_L_max = 1.0, 2000.0
t_c_min, t_c_max = -0.05, 0.05

cosine_logprob = lambda x: jnp.where(jnp.abs(x) < jnp.pi/2, jnp.log(jnp.cos(x)/2.0), -jnp.inf)
sine_logprob = lambda x: jnp.where((x >= 0.0) & (x <= jnp.pi), jnp.log(jnp.sin(x)/2.0), -jnp.inf)
uniform_logprob = lambda x, a, b: jax.scipy.stats.uniform.logpdf(x, a, b-a)
power_logprob = lambda x, n, a, b: jax.scipy.stats.beta.logpdf(x, n, n, loc=a, scale=b-a)

def logprior_fn(p):
    logprob = 0.0
    logprob += uniform_logprob(p["M_c"], M_c_min, M_c_max)
    logprob += uniform_logprob(p["q"], q_min, q_max)
    logprob += uniform_logprob(p["s1_z"], -1.0, 1.0)
    logprob += uniform_logprob(p["s2_z"], -1.0, 1.0)
    logprob += sine_logprob(p["iota"])
    logprob += power_logprob(p["d_L"], 2.0, d_L_min, d_L_max)
    logprob += uniform_logprob(p["t_c"], t_c_min, t_c_max)
    logprob += uniform_logprob(p["phase_c"], 0.0, 2 * jnp.pi)
    logprob += uniform_logprob(p["psi"], 0.0, 2 * jnp.pi)
    logprob += uniform_logprob(p["ra"], 0.0, 2 * jnp.pi)
    logprob += cosine_logprob(p["dec"])
    return logprob

# | Define the Nested Sampling algorithm
n_dims = len(columns)
n_live = 1000
n_delete = 500
num_mcmc_steps = n_dims * 3

# | Sample live points from the prior
rng_key = jax.random.PRNGKey(0)
rng_key, init_key = jax.random.split(rng_key, 2)
init_keys = jax.random.split(init_key, n_dims)
particles = {
    "M_c": jax.random.uniform(init_keys[0], (n_live,), minval=M_c_min, maxval=M_c_max),
    "q": jax.random.uniform(init_keys[1], (n_live,), minval=q_min, maxval=q_max),
    "s1_z": jax.random.uniform(init_keys[2], (n_live,), minval=-1.0, maxval=1.0),
    "s2_z": jax.random.uniform(init_keys[3], (n_live,), minval=-1.0, maxval=1.0),
    "iota": 2 * jnp.arcsin(jax.random.uniform(init_keys[4], (n_live,))**0.5),
    "d_L": jax.random.beta(init_keys[5], 2.0, 2.0, shape=(n_live,)) * (d_L_max - d_L_min) + d_L_min,
    "t_c": jax.random.uniform(init_keys[6], (n_live,), minval=t_c_min, maxval=t_c_max),
    "phase_c": jax.random.uniform(init_keys[7], (n_live,), minval=0.0, maxval=2 * jnp.pi),
    "psi": jax.random.uniform(init_keys[8], (n_live,), minval=0.0, maxval=2 * jnp.pi),
    "ra": jax.random.uniform(init_keys[9], (n_live,), minval=0.0, maxval=2 * jnp.pi),
    "dec": 2 * jnp.arcsin(jax.random.uniform(init_keys[10], (n_live,))**0.5) - jnp.pi/2.0,
}

_, ravel_fn = jax.flatten_util.ravel_pytree({k: v[0] for k, v in particles.items()})

# | Initialize the Nested Sampling algorithm
nested_sampler = blackjax.ns.adaptive.nss(
    logprior_fn=logprior_fn,
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
        lambda *args: jnp.reshape(jnp.stack(args, axis=0), 
                                  (-1,) + args[0].shape[1:]),
        *dead)
live = state.sampler_state

logL = np.concatenate((dead.logL, live.logL), dtype=float)
logL_birth = np.concatenate((dead.logL_birth, live.logL_birth), dtype=float)
data = np.concatenate([
    np.column_stack([v for v in dead.particles.values()]),
    np.column_stack([v for v in live.particles.values()])
    ], axis=0)

samples = NestedSamples(data, logL=logL, logL_birth=logL_birth, columns=columns, labels=labels)
samples.to_csv('GW.csv')
