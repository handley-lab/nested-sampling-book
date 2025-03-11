# pip install git+https://github.com/ming-256/jim

import blackjax
import blackjax.ns.adaptive
import jax
import jax.scipy.stats as stats
import jax.numpy as jnp
import numpy as np
import tqdm
from anesthetic import NestedSamples
from astropy.time import Time
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import original_likelihood as likelihood_function
from jimgw.single_event.waveform import RippleIMRPhenomD_NRTidalv2

# Define parameters class
class ParameterPrior:
    def __init__(self, name: str, label: str, prior_fn: callable, *args):
        self.name = name
        self.label = label
        self.prior_fn = prior_fn
        self.args = args

    def logprob(self, value: float) -> float:
        return self.prior_fn(value, *self.args)

# Define the prior functions
@jax.jit
def UniformPrior(x: float, min: float, max: float) -> float:    
    return stats.uniform.logpdf(x, min, max-min)

@jax.jit
def SinPrior(x):
    return jnp.log(jnp.sin(x)/2.0) + jnp.where(x < 0.0, -jnp.inf, 0.0) + jnp.where(x > jnp.pi, -jnp.inf, 0.0)

@jax.jit
def CosPrior(x):
    return jnp.log(jnp.cos(x)/2.0) + jnp.where(x < -jnp.pi/2.0, -jnp.inf, 0.0) + jnp.where(x > jnp.pi/2.0, -jnp.inf, 0.0)

@jax.jit
def BetaPrior(x, min, max):
    return stats.beta.logpdf(x, 3.0, 1.0, min, max-min)

#@jax.jit
#def FlatInLogPrior(x: float, min: float, max: float) -> float:
#    return -jnp.log(jnp.log(max / min)) - jnp.log(x)

jax.config.update('jax_enable_x64', True) 
label = 'Z_GW170817_NRTidal_General'

# | Define LIGO event data
gps = 1187008882.43
fmin = 23.0
fmax = 2048.0
duration = 128
post_trigger_duration = 2
end_time = gps + post_trigger_duration
start_time = end_time - duration
roll_off = 0.4
tukey_alpha = 2 * roll_off / duration
psd_pad = 16
psd_duration = 1024

detectors = [H1, L1, V1]
for det in detectors:
    det.load_data(gps, duration - post_trigger_duration, post_trigger_duration, fmin, fmax, psd_pad=psd_pad, psd_duration=psd_duration, tukey_alpha=tukey_alpha, gwpy_kwargs={"cache": True, "version": 2})

waveform = RippleIMRPhenomD_NRTidalv2(f_ref=fmin, use_lambda_tildes=False, no_taper=True)
frequencies = H1.frequencies
epoch = duration - post_trigger_duration
gmst = Time(gps, format="gps").sidereal_time("apparent", "greenwich").rad

# Define the priors.
parameters = [
    ParameterPrior("M_c", r"$M_c$", UniformPrior, 1.1, 1.6),
    ParameterPrior("q", r"$q$", UniformPrior, 0.125, 1.00), 
    ParameterPrior("s1_z", r"$s_{1z}$", UniformPrior, -1.0, 1.0),
    ParameterPrior("s2_z", r"$s_{2z}$", UniformPrior, -1.0, 1.0),
    ParameterPrior("iota", r"$\iota$", SinPrior),
    ParameterPrior("d_L", r"$d_L$", BetaPrior, 1.0, 1000.0),
    ParameterPrior("t_c", r"$t_c$", UniformPrior, -1.0, 1.0),
    ParameterPrior("phase_c", r"$\phi_c$", UniformPrior, 0.0, 2 * jnp.pi),
    ParameterPrior("psi", r"$\psi$", UniformPrior, 0.0, jnp.pi),
    ParameterPrior("ra", r"$\alpha$", UniformPrior, 0.0, 2 * jnp.pi),
    ParameterPrior("dec", r"$\delta$", CosPrior),
    ParameterPrior("lambda_1", r"$\Lambda_1$", UniformPrior, 0.0, 5000.0),
    ParameterPrior("lambda_2", r"$\Lambda_2$", UniformPrior, 0.0, 5000.0)
]

columns = [param.name for param in parameters]
labels = [param.label for param in parameters]

# | Define the log prior function
@jax.jit
def logprior_fn(x):
    return jnp.sum(jnp.array([param.logprob(val) for param, val in zip(parameters, x.T)]))

# | Define the likelihood function
# | Second M1 < M2 check in case sampler makes a mistake
@jax.jit
def loglikelihood_fn(x):
    params = dict(zip(columns, x.T))
    params["eta"] = params["q"] / (1 + params["q"]) ** 2
    #params["eta"] = params["q"] / (1 + params["q"]) ** 2
    params["gmst"] = gmst
    waveform_sky = waveform(frequencies, params)
    align_time = jnp.exp(-1j * 2 * jnp.pi * frequencies * (epoch + params["t_c"]))
    
    return likelihood_function(
        params,
        waveform_sky,
        detectors,
        frequencies,
        align_time,
    )

# | Define the Nested Sampling algorithm
n_dims = len(columns)
n_live = 2000
n_delete = 1000
num_mcmc_steps = n_dims * 5

# | Initialize the Nested Sampling algorithm
nested_sampler = blackjax.ns.adaptive.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    n_delete=n_delete,
    num_mcmc_steps=num_mcmc_steps,
)

@jax.jit
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = nested_sampler.step(subk, state)
    return (state, k), dead_point

# | Sample live points from the prior
# Check: Is match -> case a more efficient way of doing this or is the diff. negligible?
def sample_prior(parameter, key, n_live):
    if parameter.prior_fn == UniformPrior:
        return jax.random.uniform(key, (n_live,), minval=parameter.args[0], maxval=parameter.args[1])
    elif parameter.prior_fn == SinPrior:
        return 2 * jnp.arcsin(jax.random.uniform(key, (n_live,)) ** 0.5)
    elif parameter.prior_fn == CosPrior:
        return 2 * jnp.arcsin(jax.random.uniform(key, (n_live,)) ** 0.5) - jnp.pi / 2.0
    elif parameter.prior_fn == BetaPrior:
        return jax.random.beta(key, 3.0, 1.0, (n_live,)) * (parameter.args[1] - parameter.args[0]) + parameter.args[0]
    #elif parameter.prior_fn == FixedPrior:
    #    return jnp.full((n_live,), parameter.args[0])
#    elif parameter.prior_fn == FlatInLogPrior:
#        return parameter.args[0] * (parameter.args[1] / parameter.args[0]) ** jax.random.uniform(key, (n_live,))

rng_key = jax.random.PRNGKey(0)
rng_key, init_key = jax.random.split(rng_key, 2)
init_keys = jax.random.split(init_key, len(parameters))
initial_particles = jnp.vstack([sample_prior(param, key, n_live) for param, key in zip(parameters, init_keys)]).T

state = nested_sampler.init(initial_particles, loglikelihood_fn)

# | Run Nested Sampling
dead = []
with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    while not state.sampler_state.logZ_live - state.sampler_state.logZ < -3:
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        pbar.update(n_delete)  # Update progress bar

# | anesthetic post-processing
dead = jax.tree.map(
        lambda *args: jnp.reshape(jnp.stack(args, axis=0), 
                                  (-1,) + args[0].shape[1:]),
        *dead)
live = state.sampler_state
logL = np.concatenate((dead.logL, live.logL), dtype=float)
logL_birth = np.concatenate((dead.logL_birth, live.logL_birth), dtype=float)
data = np.concatenate((dead.particles, live.particles), dtype=float)
samples = NestedSamples(data, logL=logL, logL_birth=logL_birth, columns=columns, labels=labels)
samples.to_csv(f'{label}.csv')

logzs = samples.logZ(100)
print(f"{logzs.mean()} +- {logzs.std()}")
