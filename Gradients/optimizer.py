import math
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self):
        self.alpha = 1e-3

    def compute_step(self, f, a, x):
        Jt = jnp.transpose(jax.jacfwd(f, argnums=0)(a,x))
        return -self.alpha * jnp.matmul(Jt, f(a,x))
    
    def estimate(self, f, params):
        a = params['a']
        x = params['x']
        return a + self.compute_step(f, a, x)

class LevenbergMarquardt:
    def __init__(self):
        self.lam = 1.
        self.lambda_history = []
        self.lam_inflation_factor = 2
        self.lam_deflation_factor = 3

    def compute_step(self, f, a, x):
        # g(x) = ||f(x)||^2
        # grad_g(x) = f_prime(x)^T * f(x)
        # J = f_prime
        Jt = jnp.transpose(jax.jacfwd(f, argnums=0)(a,x))
        M = jnp.matmul(Jt, jnp.transpose(Jt)) + self.lam * jnp.eye(Jt.shape[0])
        grad_g = jnp.matmul(Jt, f(a,x))
        return -1./2 * jnp.matmul(jnp.linalg.inv(M), grad_g)

    def estimate(self, f, params):
        a = params['a']
        x = params['x']
        v = self.compute_step(f, a, x)
        f_ak = f(a + v, x)
        f_a = f(a, x)
        self.lambda_history.append(self.lam)
        if jnp.linalg.norm(f_ak) <= jnp.linalg.norm(f_a):
            ak = a + v
            self.lam /= self.lam_deflation_factor
        else:
            ak = a
            self.lam *= self.lam_inflation_factor
        return ak
    
class Optimizer:
    def __init__(self, use_lm_opt : bool = True):
        self.max_iterations = 1000
        self.convergence_threshold = 1e-8
        if use_lm_opt:
            self.opt_method = LevenbergMarquardt()
        else:
            self.opt_method = GradientDescent()

    def optimize(self, loss, a_init : jnp.array, t : jnp.array, plot_opt_results):
        residuals = np.zeros((self.max_iterations,1))
        coeffs = np.zeros((self.max_iterations, a_init.shape[0]))
        ak = a_init
        it = 0
        residual_delta = float('inf')
        while ((jnp.linalg.norm(loss(ak,t)) > self.convergence_threshold and
               residual_delta > self.convergence_threshold) and 
               it < self.max_iterations):
            residuals[it] = jnp.linalg.norm(loss(ak,t))
            coeffs[it,:] = ak
            ak = self.opt_method.estimate(loss, {'a':ak, 'x':t})
            if it > 0:
                residual_delta = jnp.linalg.norm(residuals[it]-residuals[it-1])
            it += 1

        if plot_opt_results:
            self.plot_results(it, loss, residuals, coeffs, t)

        return (coeffs[:it,:], it)
    
    def plot_results(self, num_iterations, loss, residuals, coeffs, t):
        ak = coeffs[num_iterations-1,:]
        optimization_method : str = "Gradient Descent"
        if isinstance(self.opt_method, LevenbergMarquardt):
            optimization_method = "Levenberg-Marquardt"
        fig, axes = plt.subplots(1,2)
        axes[0].plot(residuals[:num_iterations])
        axes[0].set_title("Residual Error")
        axes[0].set_xlabel("Iteration")
        if hasattr(self.opt_method, 'lambda_history'):
            axes[1].plot(self.opt_method.lambda_history[:num_iterations])
        axes[1].set_title("Lambda Dampening Factor")
        axes[1].set_xlabel("Iteration")
        plt.suptitle(f"Method: {optimization_method}")
        plt.show()
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        x = np.linspace(min(coeffs[:,1]), max(coeffs[:,1]), 50)
        y = np.linspace(min(coeffs[:,3]), max(coeffs[:,3]), 50)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                diff = loss(jnp.array([ak[0], X[i,j], ak[2], Y[i,j]]), t)
                err = jnp.dot(diff, diff)
                Z[i,j] = jnp.log(err)
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
        path_residuals = np.zeros(num_iterations)
        for i in range(num_iterations):
            diff = loss(jnp.array([ak[0], coeffs[i,1], ak[2], coeffs[i,3]]), t)
            path_residuals[i] = jnp.log(jnp.dot(diff, diff))
        ax.plot3D(coeffs[:num_iterations,1], coeffs[:num_iterations,3], path_residuals, 'r-', linewidth=2, label="Estimated Path", zorder=1)
        ax.set_xlabel("a1")
        ax.set_ylabel("a3")
        ax.set_zlabel("log-loss")
        plt.legend()
        plt.title(f"Method: {optimization_method}")
        plt.show()