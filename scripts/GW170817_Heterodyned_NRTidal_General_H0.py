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
from jimgw.single_event.detector import Detector, H1, L1, V1
from jimgw.single_event.likelihood import original_relative_binning_likelihood as relative_binning_likelihood_function
from jimgw.single_event.waveform import RippleIMRPhenomD_NRTidalv2
from flowMC.strategy.optimization import optimization_Adam
import optax
import anesthetic
import pandas as pd

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

@jax.jit
def FlatInLogPrior(x: float, min: float, max: float) -> float:
    return -jnp.log(jnp.log(max / min)) - jnp.log(x)

jax.config.update('jax_enable_x64', True) 
label = 'Z_GW170817_Heterodyned_NRTidal_General_H0'

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


waveform = RippleIMRPhenomD_NRTidalv2(f_ref=fmin, use_lambda_tildes=False, no_taper=False)
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
    ParameterPrior("t_c", r"$t_c$", UniformPrior, -0.1, 0.1),
    ParameterPrior("phase_c", r"$\phi_c$", UniformPrior, 0.0, 2 * jnp.pi),
    ParameterPrior("psi", r"$\psi$", UniformPrior, 0.0, jnp.pi),
    ParameterPrior("ra", r"$\alpha$", UniformPrior, 0.0, 2 * jnp.pi),
    ParameterPrior("dec", r"$\delta$", CosPrior),
    ParameterPrior("lambda_1", r"$\Lambda_1$", UniformPrior, 0.0, 5000.0),
    ParameterPrior("lambda_2", r"$\Lambda_2$", UniformPrior, 0.0, 5000.0),
    ParameterPrior("H_0", r"$H_0$", FlatInLogPrior, 1.0, 100.0),
    ParameterPrior("v_p", r"$v_p$", UniformPrior, -1000.0, 1000.0)
]

parameter_names = [param.name for param in parameters]
labels = [param.label for param in parameters]

# | Define the log prior function
@jax.jit
def logprior_fn(x):
    return jnp.sum(jnp.array([param.logprob(val) for param, val in zip(parameters, x.T)]))



# | SETUP HETERODYNING
# |
# |
# |
from jaxtyping import Array, Float
import numpy.typing as npt
from scipy.interpolate import interp1d

def max_phase_diff(
    f: npt.NDArray[np.floating],
    f_low: float,
    f_high: float,
    chi: Float = 1.0,
    ):
    """
    Compute the maximum phase difference between the frequencies in the array.

    Parameters
    ----------
    f: Float[Array, "n_dim"]
        Array of frequencies to be binned.
    f_low: float
        Lower frequency bound.
    f_high: float
        Upper frequency bound.
    chi: float
        Power law index.

    Returns
    -------
    Float[Array, "n_dim"]
        Maximum phase difference between the frequencies in the array.
    """

    gamma = np.arange(-5, 6, 1) / 3.0
    f = np.repeat(f[:, None], len(gamma), axis=1)
    f_star = np.repeat(f_low, len(gamma))
    f_star[gamma >= 0] = f_high
    return 2 * np.pi * chi * np.sum((f / f_star) ** gamma * np.sign(gamma), axis=1)

def compute_coefficients(data, h_ref, psd, freqs, f_bins, f_bins_center):
    A0_array = []
    A1_array = []
    B0_array = []
    B1_array = []

    df = freqs[1] - freqs[0]
    data_prod = np.array(data * h_ref.conj())
    self_prod = np.array(h_ref * h_ref.conj())
    for i in range(len(f_bins) - 1):
        f_index = np.where((freqs >= f_bins[i]) & (freqs < f_bins[i + 1]))[0]
        A0_array.append(4 * np.sum(data_prod[f_index] / psd[f_index]) * df)
        A1_array.append(
            4
            * np.sum(
                data_prod[f_index]
                / psd[f_index]
                * (freqs[f_index] - f_bins_center[i])
            )
            * df
        )
        B0_array.append(4 * np.sum(self_prod[f_index] / psd[f_index]) * df)
        B1_array.append(
            4
            * np.sum(
                self_prod[f_index]
                / psd[f_index]
                * (freqs[f_index] - f_bins_center[i])
            )
            * df
        )

    A0_array = jnp.array(A0_array)
    A1_array = jnp.array(A1_array)
    B0_array = jnp.array(B0_array)
    B1_array = jnp.array(B1_array)
    return A0_array, A1_array, B0_array, B1_array

def original_likelihood(
    params: dict[str, Float],
    h_sky: dict[str, Float[Array, " n_dim"]],
    detectors: list[Detector],
    freqs: Float[Array, " n_dim"],
    align_time: Float,
    **kwargs,
) -> Float:
    log_likelihood = 0.0
    df = freqs[1] - freqs[0]
    for detector in detectors:
        h_dec = detector.fd_response(freqs, h_sky, params) * align_time
        match_filter_SNR = (
            4 * jnp.sum((jnp.conj(h_dec) * detector.data) / detector.psd * df).real
        )
        optimal_SNR = 4 * jnp.sum(jnp.conj(h_dec) * h_dec / detector.psd * df).real
        log_likelihood += match_filter_SNR - optimal_SNR / 2

    return log_likelihood

class HeterodynedLikelihood():
    def __init__(self, detectors: list[Detector], waveform, frequencies, epoch, gmst):
        self.detectors = detectors
        self.waveform = waveform
        self.frequencies = frequencies
        self.epoch = epoch
        self.gmst = gmst
        self.n_bins = 100
        self.A0_array = {}
        self.A1_array = {}
        self.B0_array = {}
        self.B1_array = {}
        self.waveform_low_ref = {}
        self.waveform_center_ref = {}

    def make_binning_scheme(
        self, freqs: npt.NDArray[np.floating], n_bins: int, chi: float = 1
    ) -> tuple[Float[Array, " n_bins+1"], Float[Array, " n_bins"]]:

        phase_diff_array = max_phase_diff(freqs, freqs[0], freqs[-1], chi=chi)
        bin_f = interp1d(phase_diff_array, freqs)
        f_bins = np.array([])
        for i in np.linspace(phase_diff_array[0], phase_diff_array[-1], n_bins + 1):
            f_bins = np.append(f_bins, bin_f(i))
        f_bins_center = (f_bins[:-1] + f_bins[1:]) / 2
        return jnp.array(f_bins), jnp.array(f_bins_center)
        
    def maximize_likelihood(
        self,
        popsize: int = 500,
        n_steps: int = 3000,
        ):
        parameter_names = [param.name for param in parameters] + ["M_c", "eta", "gmst"]
        def y(x):
            named_params = dict(zip(parameter_names, x))
            M_1 = named_params["M_1"]
            M_2 = named_params["M_2"]
            named_params["M_c"] = (M_1 * M_2)**0.6 / (M_1 + M_2)**0.2
            named_params["eta"] = M_1 * M_2 / (M_1 + M_2)**2
            named_params["gmst"] = self.gmst
            return -self.evaluate_original(named_params)
        
        print("Starting the optimizer")

        #optimizer = optimization_Adam(
        #    n_steps=n_steps, learning_rate=0.001, noise_level=1
        #)

        optimizer = optax.adamw(
            learning_rate=0.001
        )

        initial_position = jnp.zeros((popsize, len(parameter_names))) + jnp.nan
        while not jax.tree.reduce(
            jnp.logical_and, jax.tree.map(lambda x: jnp.isfinite(x), initial_position)
        ).all():
            non_finite_index = jnp.where(
                jnp.any(
                    ~jax.tree.reduce(
                        jnp.logical_and,
                        jax.tree.map(lambda x: jnp.isfinite(x), initial_position),
                    ),
                    axis=1,
                )
            )[0]

            rng_key = jax.random.PRNGKey(0)
            rng_key, init_key = jax.random.split(rng_key, 2)
            init_keys = jax.random.split(init_key, len(parameters))
            guess = jnp.vstack([sample_prior(param, key, popsize) for param, key in zip(parameters, init_keys)]).T

            M_1, M_2 = guess[:, M_1_index], guess[:, M_2_index]
            new_M_1 = jax.lax.select(M_1 < M_2, M_2, M_1)
            new_M_2 = jax.lax.select(M_1 < M_2, M_1, M_2)
            guess = guess.at[:, M_1_index].set(new_M_1)
            guess = guess.at[:, M_2_index].set(new_M_2)
            
            M_c = (guess[:, parameter_names.index("M_1")] * guess[:, parameter_names.index("M_2")]) ** 0.6 / (guess[:, parameter_names.index("M_1")] + guess[:, parameter_names.index("M_2")]) ** 0.2
            eta = guess[:, parameter_names.index("M_1")] * guess[:, parameter_names.index("M_2")]  / (guess[:, parameter_names.index("M_1")] + guess[:, parameter_names.index("M_2")]) ** 2
            gmst2 = jnp.full((guess.shape[0],), self.gmst)
            print(gmst2.shape)
            guess = jnp.hstack([guess, M_c[:, None], eta[:, None], gmst2[:, None]])
            finite_guess = jnp.where(
                jnp.all(jax.tree.map(lambda x: jnp.isfinite(x), guess), axis=1)
            )[0]
            common_length = min(len(finite_guess), len(non_finite_index))
            initial_position = initial_position.at[
                non_finite_index[:common_length]
            ].set(guess[:common_length])

        #rng_key, optimized_positions, summary = optimizer.optimize(
        #    jax.random.PRNGKey(12094), y, initial_position
        #)

        state = optimizer.init(initial_position)
        rng = jax.random.PRNGKey(0)
        for i in range(n_steps):
            rng, step_rng = jax.random.split(rng)

            step_loss, grad = jax.value_and_grad(y)(initial_position)
            updates, state = optimizer.update(grad, state, initial_position)
            initial_position = optax.apply_updates(initial_position, updates)
        print(f"Step {i}, Loss: {step_loss}")
        print(f"final position: {initial_position}" )

        best_fit = optimized_positions[jnp.argmin(summary["final_log_prob"])]

        named_params = dict(zip(parameter_names, best_fit))
        M_1 = named_params["M_1"]
        M_2 = named_params["M_2"]
        if M_1 < M_2:
            named_params["M_1"], named_params["M_2"] = M_2, M_1
        named_params["M_c"] = (M_1 * M_2)**0.6 / (M_1 + M_2)**0.2
        named_params["eta"] = M_1 * M_2 / (M_1 + M_2)**2
        return named_params

    def evaluate_original(
        self, params: dict[str, Float]
    ) -> (
        Float
    ):
        log_likelihood = 0
        frequencies = self.frequencies
        params["gmst"] = self.gmst
        # evaluate the waveform as usual
        waveform_sky = self.waveform(frequencies, params)
        align_time = jnp.exp(
            -1j * 2 * jnp.pi * frequencies * (self.epoch + params["t_c"])
        )
        log_likelihood = original_likelihood(
            params,
            waveform_sky,
            self.detectors,
            frequencies,
            align_time
        )
        
        return log_likelihood

    def reference_state(self):
        popsize = 100
        n_steps = 2000
        #params = self.maximize_likelihood(
        #    popsize=popsize,
        #    n_steps=n_steps,
        #)
        #params = {key: float(value) for key, value in params.items()}

        samples = anesthetic.read_chains('PlotsData/Final_UniformMass_Final.csv')
        import h5py
        file_name = 'GW170817_GWTC-1.hdf5'
        data_dict = {}
        with h5py.File(file_name, "r") as hdf_file:
            dataset = hdf_file["IMRPhenomPv2NRT_lowSpin_posterior"]
            for name in dataset.dtype.names:
                data_dict[name] = np.array(dataset[name])
        param_conversion = {'M_1':'m1_detector_frame_Msun' , 'M_2':'m2_detector_frame_Msun', 'd_L':'luminosity_distance_Mpc', 'iota':'costheta_jn', 'ra':'right_ascension', 'dec':'declination', 'spin1':'spin1', 'spin2':'spin2', 'costilt1':'costilt1', 'costilt2':'costilt2', 'lambda_1':'lambda1', 'lambda_2':'lambda2'}
        columns = samples.columns 
        LVK_samples = anesthetic.MCMCSamples(columns=columns)
        for param in param_conversion.keys():
            LVK_samples[param] = data_dict[param_conversion[param]]
        LVK_samples["M_c"] = (LVK_samples["M_1"]*LVK_samples["M_2"])**0.6 / (LVK_samples["M_1"]+LVK_samples["M_2"])**0.2
        LVK_samples["q"] = LVK_samples["M_2"] / LVK_samples["M_1"]
        LVK_samples["iota"] = np.arccos(LVK_samples["iota"])
        LVK_samples["s1_z"] = LVK_samples["spin1"] * LVK_samples["costilt1"] 
        LVK_samples["s2_z"] = LVK_samples["spin2"] * LVK_samples["costilt2"]
        LVK_samples["eta"] =  LVK_samples["M_1"] * LVK_samples["M_2"] / (LVK_samples["M_1"] + LVK_samples["M_2"])**2
        means = LVK_samples.mean()
        params = means.to_dict()
        params = {k[0]: v for k, v in params.items()}
        params["t_c"] = 0.014463470602718088
        params["phase_c"] = np.pi
        params["psi"] = np.pi / 2
        params.pop('logL')
        params.pop('logL_birth')
        params.pop('nlive')

        '''
        samples["M_c"] = (samples["M_1"]*samples["M_2"])**0.6 / (samples["M_1"]+samples["M_2"])**0.2
        samples["q"] = samples["M_2"] / samples["M_1"]
        samples["eta"] =  samples["M_1"] * samples["M_2"] / (samples["M_1"] + samples["M_2"])**2
        samples.columns = pd.MultiIndex.from_tuples(
            [(col[0], col[1] if col[0] not in ['M_c', 'q'] else ('$M_c$' if col[0] == 'M_c' else '$q$')) for col in samples.columns],
            names=samples.columns.names
            )
            
        means = samples.mean()
        params = means.to_dict()
        params.pop(('logL', '$\\ln\\mathcal{L}$'))
        params.pop(('logL_birth', '$\\ln\\mathcal{L}_\\mathrm{birth}$'))
        params.pop(('nlive', '$n_\\mathrm{live}$'))
        params = {k[0]: v for k, v in params.items()}
        '''
        print(f'Optimized reference parameters: {params}')
        params["gmst"] = self.gmst
        h_sky = waveform(self.frequencies, params)
        # Get the grid of the relative binning scheme (contains the final endpoint)
        # and the center points
        freq_grid, self.freq_grid_center = self.make_binning_scheme(
            np.array(self.frequencies), self.n_bins
            )
        self.freq_grid_low = freq_grid[:-1]
        if jnp.isclose(params["eta"], 0.25):
            params["eta"] = 0.249995
        # Get frequency masks to be applied, for both original
        # and heterodyne frequency grid
        h_amp = jnp.sum(
            jnp.array([jnp.abs(h_sky[key]) for key in h_sky.keys()]), axis=0
        )
        f_valid = self.frequencies[jnp.where(h_amp > 0)[0]]
        f_max = jnp.max(f_valid)
        f_min = jnp.min(f_valid)

        mask_heterodyne_grid = jnp.where((freq_grid <= f_max) & (freq_grid >= f_min))[0]
        mask_heterodyne_low = jnp.where(
            (self.freq_grid_low <= f_max) & (self.freq_grid_low >= f_min)
        )[0]
        mask_heterodyne_center = jnp.where(
            (self.freq_grid_center <= f_max) & (self.freq_grid_center >= f_min)
        )[0]
        freq_grid = freq_grid[mask_heterodyne_grid]
        self.freq_grid_low = self.freq_grid_low[mask_heterodyne_low]
        self.freq_grid_center = self.freq_grid_center[mask_heterodyne_center]

        # Assure frequency grids have same length
        if len(self.freq_grid_low) > len(self.freq_grid_center):
            self.freq_grid_low = self.freq_grid_low[: len(self.freq_grid_center)]

        h_sky_low = self.waveform(self.freq_grid_low, params)
        h_sky_center = self.waveform(self.freq_grid_center, params)
        # Get phase shifts to align time of coalescence
        align_time = jnp.exp(
            -1j
            * 2
            * jnp.pi
            * self.frequencies
            * (self.epoch + params["t_c"])
        )
        align_time_low = jnp.exp(
            -1j
            * 2
            * jnp.pi
            * self.freq_grid_low
            * (self.epoch + params["t_c"])
        )
        align_time_center = jnp.exp(
            -1j
            * 2
            * jnp.pi
            * self.freq_grid_center
            * (self.epoch + params["t_c"])
        )

        for detector in self.detectors:
            waveform_ref = (
                detector.fd_response(self.frequencies, h_sky, params)
                * align_time
            )
            self.waveform_low_ref[detector.name] = (
                detector.fd_response(self.freq_grid_low, h_sky_low, params)
                * align_time_low
            )
            self.waveform_center_ref[detector.name] = (
                detector.fd_response(
                    self.freq_grid_center, h_sky_center, params
                )
                * align_time_center
            )
            A0, A1, B0, B1 = compute_coefficients(
                detector.data,
                waveform_ref,
                detector.psd,
                self.frequencies,
                freq_grid,
                self.freq_grid_center,
            )
            self.A0_array[detector.name] = A0[mask_heterodyne_center]
            self.A1_array[detector.name] = A1[mask_heterodyne_center]
            self.B0_array[detector.name] = B0[mask_heterodyne_center]
            self.B1_array[detector.name] = B1[mask_heterodyne_center]

    def evaluate(self, params: dict[str, Float]) -> Float:
        frequencies_low = self.freq_grid_low
        frequencies_center = self.freq_grid_center
        params["gmst"] = self.gmst
        # evaluate the waveforms as usual
        waveform_sky_low = self.waveform(frequencies_low, params)
        waveform_sky_center = self.waveform(frequencies_center, params)
        align_time_low = jnp.exp(
            -1j * 2 * jnp.pi * frequencies_low * (self.epoch + params["t_c"])
        )
        align_time_center = jnp.exp(
            -1j * 2 * jnp.pi * frequencies_center * (self.epoch + params["t_c"])
        )
        return relative_binning_likelihood_function(
            params,
            self.A0_array,
            self.A1_array,
            self.B0_array,
            self.B1_array,
            waveform_sky_low,
            waveform_sky_center,
            self.waveform_low_ref,
            self.waveform_center_ref,
            self.detectors,
            frequencies_low,
            frequencies_center,
            align_time_low,
            align_time_center
        )
# |
# |
# |

likelihood_function = HeterodynedLikelihood(detectors, waveform, frequencies, epoch, gmst)

# | Define the likelihood function
# | Extra check on M1 < M2 to catch points that slipped through
@jax.jit
def loglikelihood_fn(x):
    params = dict(zip(parameter_names, x.T))
    params["eta"] = params["q"] / (1 + params["q"]) ** 2

    ll_vr = jax.scipy.stats.norm.logpdf(3327, params["v_p"] + params["H_0"] * params["d_L"], 72)
    ll_vp = jax.scipy.stats.norm.logpdf(310, params["v_p"], 150)

    return likelihood_function.evaluate(params) + ll_vr + ll_vp

# | Define the Nested Sampling algorithm
n_dims = len(parameter_names)
n_live = 10000
n_delete = 5000
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
    elif parameter.prior_fn == FlatInLogPrior:
        return parameter.args[0] * (parameter.args[1] / parameter.args[0]) ** jax.random.uniform(key, (n_live,))

rng_key = jax.random.PRNGKey(0)
rng_key, init_key = jax.random.split(rng_key, 2)
init_keys = jax.random.split(init_key, len(parameters))
initial_particles = jnp.vstack([sample_prior(param, key, n_live) for param, key in zip(parameters, init_keys)]).T

likelihood_function.reference_state()
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
samples = NestedSamples(data, logL=logL, logL_birth=logL_birth, columns=parameter_names, labels=labels)
samples.to_csv(f'{label}.csv')

logzs = samples.logZ(100)
print(f"{logzs.mean()} +- {logzs.std()}")
