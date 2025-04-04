#!/usr/bin/env python
# coding: utf-8

import jax
import jax.numpy as jnp
import tinygp
import blackjax
import matplotlib.pyplot as plt
import tqdm
import tensorflow_probability.substrates.jax as tfp
from blackjax.ns.utils import finalise, sample
from functools import partial
import os
import matplotlib.pyplot as plt
from anesthetic import MCMCSamples

os.makedirs("figures", exist_ok=True)

tfd = tfp.distributions
tfb = tfp.bijectors
rng_key = jax.random.PRNGKey(0)


X = jnp.linspace(start=0, stop=10, num=500)
true_weights = jnp.array([1.5, 1.0])
true_freqs = jnp.array([0.1, 1.0])
data_set_size = 100
noise_level = 0.2


def model(theta, x):
    weights = theta["weight"]
    freqs = theta["freq"]
    return jnp.sum(
        weights * jnp.sin(2 * jnp.pi * x[..., None] * jnp.array(freqs)),
        axis=-1,
    )


rng_key, noise_key = jax.random.split(rng_key)
y = model({"weight": true_weights, "freq": true_freqs}, x=X)

rng_key, data_key = jax.random.split(rng_key)
training_indices = jax.random.choice(
    data_key, jnp.arange(y.size), shape=(data_set_size,), replace=False
)
X_train, y_train = X[training_indices], y[training_indices]

y_train = y_train + jax.random.normal(noise_key, y_train.shape) * noise_level

fig, ax = plt.subplots(figsize=(8, 6))

# Plot on the axis object instead of using plt directly
ax.plot(X, y, label=r"$f(x)$", linestyle="dotted")
ax.errorbar(
    X_train,
    y_train,
    yerr=noise_level,
    fmt="o",
    capsize=3,
    label="Noisy observations",
    color="black",
)
ax.legend()
ax.set_xlabel("$x$")
ax.set_ylabel("$f(x)$")
ax.set_title("True generative process")

# Save the figure
fig.savefig(
    os.path.join("figures", "generative_process.png"),
    dpi=300,
    bbox_inches="tight",
)


fundamental_freq = 1 / (X.max() - X.min())
sampling_freq = X.shape[0] * fundamental_freq


@jax.jit
def loglikelihood(theta):
    n = len(y_train)
    return -0.5 * jnp.sum(
        (model(theta, x=X_train) - y_train) ** 2 / theta["noise"] ** 2
    ) #- 0.5 * n * jnp.log(2 * jnp.pi * theta["noise"] ** 2)


def build_prior(n_components):
    # Define individual priors
    weight_prior = tfd.Normal(
        loc=jnp.zeros(n_components),
        scale=jnp.ones(n_components),
    )

    freq_prior = tfd.Uniform(
        low=jnp.ones(n_components) * fundamental_freq,
        high=jnp.ones(n_components) * sampling_freq / 2,
    )

    noise_prior = tfd.Uniform(
        low=0.0,
        high=1.0,
    )

    # Define joint distribution for sampling
    prior = tfd.JointDistributionNamed(
        {
            "weight": weight_prior,
            "freq": freq_prior,
            "noise": noise_prior,
        }
    )

    # Manual log probability function
    def log_prob(params):
        """Calculate the log probability of the parameters.

        Args:
            params: Dictionary with keys 'weight', 'scale', 'freq', and 'noise'

        Returns:
            Total log probability
        """
        total_log_prob = 0.0
        total_log_prob += weight_prior.log_prob(params["weight"]).sum()
        total_log_prob += freq_prior.log_prob(params["freq"]).sum()
        total_log_prob += noise_prior.log_prob(params["noise"]).sum()
        return total_log_prob

    return prior, log_prob


prior, log_prob = build_prior(2)

loglikelihood(prior.sample(seed=jax.random.PRNGKey(0)))


test_sample, ravel_fn = jax.flatten_util.ravel_pytree(
    prior.sample(seed=jax.random.PRNGKey(0))
)


n_dims = test_sample.shape[0]

n_live = 1000
n_delete = 100

num_mcmc_steps = n_dims * 3


def integrate(nested_sampler, rng_key, sort=False):
    rng_key, init_key = jax.random.split(rng_key, 2)
    particles = prior.sample(seed=init_key, sample_shape=(n_live,))
    if sort:
        idx = jnp.argsort(particles["freq"])
        particles["freq"] = jnp.take_along_axis(particles["freq"], idx, -1)
        particles["weight"] = jnp.take_along_axis(particles["weight"], idx, -1)
    state = nested_sampler.init(
        particles,
        loglikelihood,
    )

    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = nested_sampler.step(subk, state)
        return (state, k), dead_point

    dead = []
    with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
        while (
            not state.sampler_state.logZ_live - state.sampler_state.logZ < -3
        ):
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            # (state, rng_key), dead_info = multi_steps((state, rng_key), None)

            dead.append(dead_info)
            pbar.update(n_delete)
    res = (state, finalise(state, dead))
    print("total evals:", res[1].update_info.evals.sum())
    return res


def plot(rng_key, final):
    rng_key, sample_key = jax.random.split(rng_key)
    posterior_samples = sample(sample_key, final, 100)
    f, a = plt.subplots()
    a.plot(X, jax.vmap(partial(model, x=X))(posterior_samples).T, color="C0")
    a.errorbar(
        X_train,
        y_train,
        yerr=noise_level,
        fmt="none",
        c="black",
        label="data",
        zorder=10,
    )
    a.scatter(X_train, y_train, c="black", label="data", zorder=10)
    a.set_xlabel("$x$")
    a.set_ylabel("$f(x)$")
    a.set_title(
        "Posterior samples, $N = {}$".format(
            posterior_samples["freq"].shape[1]
        )
    )
    f.savefig(
        "figures/f_x_{}.png".format(posterior_samples["freq"].shape[1]),
        dpi=300,
    )
    plt.close()
    a = MCMCSamples(
        posterior_samples["freq"],
        columns=[
            r"$\mu_{}$".format(i)
            for i in range(posterior_samples["freq"].shape[1])
        ],
    ).plot_2d()
    plt.savefig(
        "figures/post_{}.png".format(posterior_samples["freq"].shape[1]),
        dpi=300,
    )


def wrapped_stepper(x, n, t):
    y = jax.tree_map(lambda x, n: x + t * n, x, n)
    idx = jnp.argsort(y["freq"])
    y["freq"] = jnp.take_along_axis(y["freq"], idx, -1)
    y["weight"] = jnp.take_along_axis(y["weight"], idx, -1)
    return y


##################################################################
# Integrate Model 1
##################################################################
prior, log_prob = build_prior(1)
test_sample, ravel_fn = jax.flatten_util.ravel_pytree(
    prior.sample(seed=jax.random.PRNGKey(0))
)
nested_sampler = blackjax.ns.adaptive.nss(
    logprior_fn=log_prob,
    loglikelihood_fn=loglikelihood,
    n_delete=n_delete,
    num_mcmc_steps=num_mcmc_steps,
    ravel_fn=ravel_fn,
    stepper=wrapped_stepper,
)

state_model_1, final_1 = integrate(nested_sampler, rng_key, sort=True)
initial_particles_bf = sample(rng_key, final_1, 1000)
print(
    f"N={final_1.particles["freq"].shape[1]} components, logZ={state_model_1.sampler_state.logZ:.2f}"
)
plot(rng_key, final_1)

##################################################################
# Integrate Model 2
##################################################################

prior, log_prob = build_prior(2)
test_sample, ravel_fn = jax.flatten_util.ravel_pytree(
    prior.sample(seed=jax.random.PRNGKey(0))
)
nested_sampler = blackjax.ns.adaptive.nss(
    logprior_fn=log_prob,
    loglikelihood_fn=loglikelihood,
    n_delete=n_delete,
    num_mcmc_steps=num_mcmc_steps,
    ravel_fn=ravel_fn,
    stepper=wrapped_stepper,
)
state_model_2, final_2 = integrate(nested_sampler, rng_key, sort=True)
print(
    f"N={final_2.particles["freq"].shape[1]} components, logZ={state_model_2.sampler_state.logZ:.2f}"
)
plot(rng_key, final_2)
##################################################################

print(
    f"Bayes Factor: {state_model_2.sampler_state.logZ - state_model_1.sampler_state.logZ:.2f}"
)

##################################################################
# Now lets try to estimate this bayes factor directly

prior, log_prob = build_prior(2)
test_sample, ravel_fn = jax.flatten_util.ravel_pytree(
    prior.sample(seed=jax.random.PRNGKey(0))
)


def new_log_prior(theta):
    original_prior = log_prob(theta)
    # filter out the l_1 samples from theta
    theta_1 = {
        "weight": theta["weight"][:1],
        "freq": theta["freq"][:1],
        "noise": theta["noise"],
    }
    l_1 = loglikelihood(theta_1)
    return original_prior + l_1


def new_loglikelihood(theta):
    # filter out the l_1 samples from theta
    theta_1 = {
        "weight": theta["weight"][:1],
        "freq": theta["freq"][:1],
        "noise": theta["noise"],
    }
    l_1 = loglikelihood(theta_1)
    return loglikelihood(theta) - l_1


nested_sampler = blackjax.ns.adaptive.nss(
    logprior_fn=new_log_prior,
    loglikelihood_fn=new_loglikelihood,
    n_delete=n_delete,
    num_mcmc_steps=num_mcmc_steps,
    ravel_fn=ravel_fn,
    # stepper=wrapped_stepper,
)

prior_1, _ = build_prior(1)
augment_points = prior_1.sample(seed=rng_key, sample_shape=(n_live,))

initial_particles_bf["weight"] = jnp.concatenate(
    [initial_particles_bf["weight"], augment_points["weight"]], axis=-1
)
initial_particles_bf["freq"] = jnp.concatenate(
    [initial_particles_bf["freq"], augment_points["freq"]], axis=-1
)
# initial_particles_bf["noise"] = augment_points["noise"]


state = nested_sampler.init(
    initial_particles_bf,
    new_loglikelihood,
)


@jax.jit
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = nested_sampler.step(subk, state)
    return (state, k), dead_point


dead = []

with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    while not state.sampler_state.logZ_live - state.sampler_state.logZ < -3:
        (state, rng_key), dead_info = one_step((state, rng_key), None)

        dead.append(dead_info)
        pbar.update(n_delete)

fs = finalise(state, dead)
print("total evals:", fs.update_info.evals.sum())
print("logZ:", state.sampler_state.logZ)
from anesthetic import NestedSamples

samples = NestedSamples(
    data=jnp.concatenate(
        [fs.particles["freq"], fs.particles["noise"][..., None]], axis=-1
    ),
    logL=fs.logL,
    logL_birth=fs.logL_birth,
)
samples.to_csv("BF.csv")
