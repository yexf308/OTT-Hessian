import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import time
import timeit
from jax.example_libraries import optimizers as jax_opt


import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".50"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

import ott
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

import numpy as np
import SinkhornHessian
import util

import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1

font = {'size'   : 20}
matplotlib.rc('font', **font)

if __name__ == '__main__':
    dim = 2
    n_span       = [10, 20, 30, 40, 80, 120, 180]
    epsilon_span = np.array([5e-3,  5e-2,  5e-1])
    K = 100
    svd_thr = 1e-10

    SH = SinkhornHessian.SinkhornHessian(svd_thr)
    

    print("----Testing different epsilon, n and methods to calculate Hessian---")
    error_a        = np.ones([len(n_span), len(epsilon_span), K]) * np.nan
    error_a_no_reg = np.ones([len(n_span), len(epsilon_span), K]) * np.nan
    error_u        = np.ones([len(n_span), len(epsilon_span), K]) * np.nan
    error_i        = np.ones([len(n_span), len(epsilon_span), K]) * np.nan


    for i, n in enumerate(n_span):
        print(f"n: {n}")
        hess_analytic_𝜀_jit               = jax.jit(SH.hess_loss_analytical)
        hess_analytic_𝜀_jit_no_reg        = jax.jit(SH.hess_loss_analytical_no_reg)
        hess_unroll_𝜀_jit                 = jax.jit(SH.hess_loss_unroll)
        hess_implicit_𝜀_jit               = jax.jit(SH.hess_loss_implicit)

        for k in range(K):

            
            mu, nv, x, y = util.sample_points_uniform(n, n, dim, k)
            y = x
            epsilon_base = pointcloud.PointCloud(x, y).mean_cost_matrix
            threshold_n = 0.01 / (n**0.33)

            I_d = jnp.eye(dim)
            true_sum =  2 * mu[:, None, None] * I_d[None, :, :]
            true_sum_ravel = true_sum.ravel()
            for j, epsilon_scale in enumerate(epsilon_span):
                epsilon =  epsilon_scale * epsilon_base
                print("n:",n,", 𝜀_scale:", epsilon_scale, f", 𝜀: {epsilon:.5f}", f", thr.: {threshold_n:.5f}",end=" |",)
                hess_analytic_𝜀                   = hess_analytic_𝜀_jit(x, y, mu, nv, epsilon, threshold_n)
                hess_analytic_𝜀_sum               = jnp.sum(hess_analytic_𝜀, axis=2)
                error_a[i,j,k]                    = jnp.linalg.norm(true_sum_ravel - hess_analytic_𝜀_sum.ravel())

                hess_analytic_𝜀_no_reg           = hess_analytic_𝜀_jit_no_reg(x, y, mu, nv, epsilon, threshold_n)
                hess_analytic_𝜀_no_reg_sum       = jnp.sum(hess_analytic_𝜀_no_reg, axis=2)
                error_a_no_reg[i,j,k]            = jnp.linalg.norm(true_sum_ravel - hess_analytic_𝜀_no_reg_sum.ravel())

                hess_unroll_𝜀                    = hess_unroll_𝜀_jit(x, y, mu, nv, epsilon, threshold_n)
                #hess_unroll_𝜀 = ott_unroll_hessian(x, y, mu, nv, epsilon, threshold_n)
                hess_unroll_𝜀_sum                = jnp.sum(hess_unroll_𝜀, axis=2)
                error_u[i,j,k]                   = jnp.linalg.norm(true_sum_ravel - hess_unroll_𝜀_sum.ravel())

                try:
                    hess_implicit_𝜀                       = hess_implicit_𝜀_jit(x, y, mu, nv, epsilon, threshold_n)
                    hess_implicit_𝜀_sum                   = jnp.sum(hess_implicit_𝜀, axis=2)
                    error_i[i,j,k]                        = jnp.linalg.norm(true_sum_ravel - hess_implicit_𝜀_sum.ravel())

                except :
                    print("Error in implicit method")
                    error_i[i,j,k] = np.nan

                print()
                
        

    np.savez('error_uniform.npz', error_a=error_a,  error_i = error_i, error_a_no_reg=error_a_no_reg )            

    list_legend = []
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(n_span, np.nanmean(error_a[:,0,:],axis=1),marker = "o", color="red",  markersize=10, linestyle="-",
            markeredgecolor="k",
            lw=3,label="Analytical $𝜀$=0.005" )

    ax.plot(n_span, np.nanmean(error_a[:,1,:],axis=1),marker = "d", color="red",  markersize=10, linestyle="--",
            markeredgecolor="k",
            lw=3,label="Analytical $𝜀$=0.05" )
    ax.plot(n_span, np.nanmean(error_a[:,2,:],axis=1),marker = "p", color="red",  markersize=10, linestyle=":",
            markeredgecolor="k",
            lw=3,label="Analytical $𝜀$=0.5" )

    ax.plot(n_span, np.nanmean(error_i[:,0,:],axis=1),marker = "o", color="blue",  markersize=10, linestyle="-",
            markeredgecolor="k",
            lw=3,label="Implicit $𝜀$=0.005" )

    ax.plot(n_span, np.nanmean(error_i[:,1,:],axis=1),marker = "d", color="blue",  markersize=10, linestyle="--",
            markeredgecolor="k",
            lw=3,label="Implicit $𝜀$=0.05" )
    ax.plot(n_span, np.nanmean(error_i[:,2,:],axis=1),marker = "p", color="blue",  markersize=10, linestyle=":",
            markeredgecolor="k",
            lw=3,label="Implicit $𝜀$=0.5" )


    ax.plot(n_span, np.nanmean(error_u[:,0,:],axis=1),marker = "o", color="purple",  markersize=10, linestyle="-",
            markeredgecolor="k",
            lw=3,label="Unroll $𝜀$=0.005" )

    ax.plot(n_span, np.nanmean(error_u[:,1,:],axis=1),marker = "d", color="purple",  markersize=10, linestyle="--",
            markeredgecolor="k",
            lw=3,label="Unroll $𝜀$=0.05" )
    ax.plot(n_span, np.nanmean(error_u[:,2,:],axis=1),marker = "p", color="purple",  markersize=10, linestyle=":",
            markeredgecolor="k",
            lw=3,label="Unroll $𝜀$=0.5" )

    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.xlabel("$n$")
    plt.ylabel("Marginal error of Hessian")
    ax.set_xticks([20,30, 40, 80, 120, 180, 270])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid()
    ax.legend(loc=(0.1,0.15),ncol = len(ax.lines)/3 )
    plt.title("Uniformly distributed point cloud in unit square")
    plt.savefig("accuracy_uniform.pdf", format="pdf", bbox_inches="tight")
    plt.show()



