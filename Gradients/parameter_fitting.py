import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random

from optimizer import LevenbergMarquardt

class DataGenerator:
    def __init__(self, coefficients : jnp.array):
        self.times : jnp.array
        self.values : jnp.array
        self.measurements : jnp.array
        self.coefficients = coefficients
        self.noise_mu = 0.0
        self.noise_sigma = 0.2
 
    def generate(self):
        self.times, self.values = self.generate_data()

    def collect_observations(self):
        key = random.PRNGKey(32)
        _, subkey = random.split(key)
        noise = random.normal(subkey, shape=self.values.shape)
        scaled_noise = self.noise_mu + self.noise_sigma * noise
        self.measurements = self.values + scaled_noise
        return self.measurements
    
    def generate_data(self):
        t = jnp.linspace(0, 100, 500)
        return (t, model(self.coefficients, t))
    
def plot_data(times, values, measurements, parameter_estimates):
    fig, axes = plt.subplots(1,2)
    axes[0].plot(times, values, label="Ground Truth", color="green")
    axes[0].scatter(times, measurements, label="Measurements", s=2, color='red')
    axes[0].plot(times, model(parameter_estimates, times), label="Estimates", color='blue')
    axes[0].set_xlabel('Time')
    axes[1].hist(values - measurements, label="Measurement Errors")
    axes[1].hist(values - model(parameter_estimates, times), label="Residual Errors")
    for ax in axes:
        ax.legend()
    plt.tight_layout()
    plt.show()

def model(a: jnp.array, t : jnp.array):
    '''
    Sample data generated from function:
        F(t) = a1 exp(-t/a2) + a3 t exp(-t/a4)
    '''
    return a[0] * jnp.exp(-t/a[1]) + a[2] * t * jnp.exp(-t / a[3])

def run_model_fitting(subkey):
    a = jnp.array([3.9, 2.8, 2.0, 10.])
    generator = DataGenerator(a)
    generator.generate()
    observations = generator.collect_observations()

    @jax.jit
    def loss(a, t):
        return observations - model(a,t)

    def fit_model():
        # Do the initial estimates need to be positive?
        initial_coeffs = jnp.abs(random.normal(subkey, shape=(4,)))
        print(f"True coefficients: {a} / initial guess: {initial_coeffs}")
        t = jnp.linspace(0, 100, 500)
        lm_opt = LevenbergMarquardt()
        coeffs = lm_opt.optimize(loss, initial_coeffs, t, plot_opt_results=True)
        plot_data(generator.times, generator.values, generator.measurements, coeffs[-1,:])
        
    fit_model()

def main():
    subkey = random.PRNGKey(234)
    for i in range(3):
        _, subkey = random.split(subkey)
        run_model_fitting(subkey)

if __name__ == '__main__':
    main()
