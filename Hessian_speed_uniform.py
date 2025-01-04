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
    epsilon_span = np.array([5e-3,  5e-2,  5e-1])
    K =11
    svd_thr = 1e-10
    SH = SinkhornHessian.SinkhornHessian(svd_thr)

    """Analytical Hessian"""
    n_span       = [10, 20, 30, 40, 80, 120, 180, 270, 400, 800, 1600, 3200,6400]
    result_t_analytic = np.ones((len(n_span), len(epsilon_span), K)) * np.nan
    hess_analytic_𝜀_jit               = jax.jit(SH.hess_loss_analytical)

    for k in range(K):
        for i, n in enumerate(n_span):
            print()
            mu, nv, x, y = util.sample_points_uniform(n, n, dim, k)
            y = x
            epsilon_base = pointcloud.PointCloud(x, y).mean_cost_matrix
            threshold_n = 0.05 / (n**0.33)

            for j, epsilon_scale in enumerate(epsilon_span):
                epsilon =  epsilon_scale * epsilon_base
                print("n:",n,", 𝜀_scale:", epsilon_scale, f", 𝜀: {epsilon:.5f}", f", thr.: {threshold_n:.5f}",end=" |",)

                start_time = time.perf_counter()
                hess_analytic_𝜀_jit(x, y, mu, nv, epsilon, threshold_n)
                end_time = time.perf_counter()
                result_t_analytic[i,j,k] = end_time - start_time

    result_t_analytic_final = result_t_analytic[:,:,1:]

    """Unroll Hessian"""
    n_span       = [10, 20, 30, 40, 80, 120, 180]
    result_t_unroll = np.ones((len(n_span), len(epsilon_span), K)) * np.nan
    hess_unroll_𝜀_jit                 = jax.jit(SH.hess_loss_unroll)

    for k in range(K):
        for i, n in enumerate(n_span):
            print()
            mu, nv, x, y = util.sample_points_uniform(n, n, dim, k)
            y = x
            epsilon_base = pointcloud.PointCloud(x, y).mean_cost_matrix
            threshold_n = 0.05 / (n**0.33)

            for j, epsilon_scale in enumerate(epsilon_span):
                epsilon =  epsilon_scale * epsilon_base
                print("n:",n,", 𝜀_scale:", epsilon_scale, f", 𝜀: {epsilon:.5f}", f", thr.: {threshold_n:.5f}",end=" |",)

                start_time = time.perf_counter()
                hess_unroll_𝜀_jit(x, y, mu, nv, epsilon, threshold_n)
                end_time = time.perf_counter()
                result_t_unroll[i,j,k] = end_time - start_time

    result_t_unroll_final = result_t_unroll[:,:,1:]


    """Implicit Hessian"""

    n_span       = [10, 20, 30, 40, 80, 120, 180, 270, 400]
    result_t_implicit = np.ones((len(n_span), len(epsilon_span), K)) * np.nan
    hess_implicit_𝜀_jit               = jax.jit(SH.hess_loss_implicit)

    for k in range(K):
        for i, n in enumerate(n_span):
            print()
            mu, nv, x, y = util.sample_points_uniform(n, n, dim, k)
            y = x
            epsilon_base = pointcloud.PointCloud(x, y).mean_cost_matrix
            threshold_n = 0.05 / (n**0.33)

            for j, epsilon_scale in enumerate(epsilon_span):
                epsilon =  epsilon_scale * epsilon_base
                print("n:",n,", 𝜀_scale:", epsilon_scale, f", 𝜀: {epsilon:.5f}", f", thr.: {threshold_n:.5f}",end=" |",)
                try:
                    start_time = time.perf_counter()
                    hess_implicit_𝜀_jit(x, y, mu, nv, epsilon, threshold_n)
                    end_time = time.perf_counter()
                    result_t_implicit[i,j,k] = end_time - start_time
                except:
                    print("Error in implicit method")
                    pass

    result_t_implicit_final = result_t_implicit[:,:,1:]

    np.savez('t_uniform.npz', t1=result_t_analytic_final, t2=result_t_implicit_final, t3=result_t_unroll_final)



    list_legend = []
    fig, ax = plt.subplots(figsize=(12, 8))

    n_span       = [10, 20, 30, 40, 80, 120, 180]

    ax.plot(n_span, np.nanmean(result_t_unroll_final[:len(n_span),0,:],axis=1),marker = "o", color="purple",  markersize=12, linestyle="-",
            markeredgecolor="k",
            lw=3,label="Unroll $𝜀$=0.005" )
    ax.plot(n_span, np.nanmean(result_t_unroll_final[:len(n_span),1,:],axis=1),marker = "d", color="purple",  markersize=12, linestyle="--",
            markeredgecolor="k",
            lw=3,label="Unroll $𝜀$=0.05" )


    n_span       = [10, 20, 30, 40, 80, 120, 180, 270, 400]

    ax.plot(n_span, np.nanmean(result_t_implicit_final[:len(n_span),0,:],axis=1),marker = "o", color="red",  markersize=12, linestyle="-",
            markeredgecolor="k",
            lw=3,label="Implicit $𝜀$=0.005" )
    ax.plot(n_span, np.nanmean(result_t_implicit_final[:len(n_span),1,:],axis=1),marker = "d", color="red",  markersize=12, linestyle="--",
            markeredgecolor="k",
            lw=3,label="Implicit $𝜀$=0.05" )





    n_span       = [10, 20, 30, 40, 80, 120, 180, 270, 400, 800, 1600, 3200,6400]
    ax.plot(n_span, np.nanmean(result_t_analytic_final[:len(n_span),0,:],axis=1),marker = "o", color="blue",  markersize=12,linestyle="-",
            markeredgecolor="k",
            lw=3,label="Analytic $𝜀$=0.005" )
    ax.plot(n_span, np.nanmean(result_t_analytic_final[:len(n_span),1,:],axis=1),marker = "d", color="blue",  markersize=12,linestyle="--",
            markeredgecolor="k",
            lw=3,label="Analytic $𝜀$=0.05" )






    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.xlabel("$N$", fontsize = 28)
    ax.set_xticks([20, 40, 100, 200, 400, 800, 1600, 3200,6400])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.ylabel("Average Execution Time", fontsize = 25)
    plt.title("Uniformly distributed point cloud in unit square")
    ax.legend()
    plt.savefig("speed_uniform.pdf", format="pdf", bbox_inches="tight")
    plt.show()




 