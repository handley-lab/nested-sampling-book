import blackjax
import blackjax.ns.adaptive
import jax
import jax.scipy.stats as stats
import jax.numpy as jnp
import numpy as np
import tqdm
from anesthetic import NestedSamples
from astropy.time import Time
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.likelihood import original_likelihood as likelihood_function
from jimgw.single_event.waveform import RippleIMRPhenomD

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

jax.config.update('jax_enable_x64', True) 
label = 'Final_UniformMass_constrained_prior_2'

# | Define LIGO event data
gps = 1126259462.4
fmin = 20.0
fmax = 1024.0
duration = 4
post_trigger_duration = 2
end_time = gps + post_trigger_duration
start_time = end_time - duration
roll_off = 0.4
tukey_alpha = 2 * roll_off / duration
psd_pad = 16

detectors = [H1, L1]
for det in detectors:
    det.load_data(gps, duration - post_trigger_duration, post_trigger_duration, fmin, fmax, psd_pad=psd_pad, tukey_alpha=tukey_alpha)

waveform = RippleIMRPhenomD(f_ref=fmin)
frequencies = H1.frequencies
epoch = duration - post_trigger_duration
gmst = Time(gps, format="gps").sidereal_time("apparent", "greenwich").rad

# Define the priors.
parameters = [
    ParameterPrior("M_1", r"$M_1$", UniformPrior, 10.0, 80.0),
    ParameterPrior("M_2", r"$M_2$", UniformPrior, 10.0, 80.0), 
    ParameterPrior("s1_z", r"$s_{1z}$", UniformPrior, -1.0, 1.0),
    ParameterPrior("s2_z", r"$s_{2z}$", UniformPrior, -1.0, 1.0),
    ParameterPrior("iota", r"$\iota$", SinPrior),
    ParameterPrior("d_L", r"$d_L$", BetaPrior, 1.0, 2000.0),
    ParameterPrior("t_c", r"$t_c$", UniformPrior, -0.05, 0.05),
    ParameterPrior("phase_c", r"$\phi_c$", UniformPrior, 0.0, 2 * jnp.pi),
    ParameterPrior("psi", r"$\psi$", UniformPrior, 0.0, jnp.pi),
    ParameterPrior("ra", r"$\alpha$", UniformPrior, 0.0, 2 * jnp.pi),
    ParameterPrior("dec", r"$\delta$", CosPrior)  
]

columns = [param.name for param in parameters]
labels = [param.label for param in parameters]

# | Define the log prior function
def logprior_fn(x):
    param_dict = dict(zip([param.name for param in parameters], x.T))
    M_1 = param_dict["M_1"]
    M_2 = param_dict["M_2"]
    logprob_mass = jnp.where(M_1 < M_2, -jnp.inf, parameters[0].logprob(M_1) + parameters[1].logprob(M_2))
    logprobs = jnp.array([logprob_mass] + [param.logprob(value) for param, value in zip(parameters[2:], x.T[2:])])
    return jnp.sum(logprobs) + jnp.log(2)


# | Define the likelihood function
# | Second M1 < M2 check in case sampler makes a mistake
@jax.jit
def loglikelihood_fn(x):
    params = dict(zip([param.name for param in parameters], x.T))
    M_1 = params["M_1"]
    M_2 = params["M_2"]
    def compute_likelihood(params, M_1, M_2):
        params["M_c"] = (M_1 * M_2)**0.6 / (M_1 + M_2)**0.2
        params["eta"] = M_1 * M_2 / (M_1 + M_2)**2
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
    return jax.lax.cond(
        M_1 < M_2,
        lambda _: -jnp.inf,
        lambda _: compute_likelihood(params, M_1, M_2),
        operand=None
    )

# | Define the Nested Sampling algorithm
n_dims = len(columns)
n_live = 4000
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
