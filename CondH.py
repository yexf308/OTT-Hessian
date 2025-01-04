import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

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
    

    color_span   = ['r','b','g','y']
    marker_span  = ['o','s','d','p']
    ##################################################
    """Case 1: 2D circle example: lambdas vs epsilon """

    n_span       = [ 50,150, 500, 1500]

    epsilon_span = np.array([7e-7, 1e-6, 2e-6, 5e-6,7e-6, 9e-6, 1e-5, 1.5e-5, 2e-5, 3e-5, 4e-5, 5e-5, 7e-5, 9e-5,1e-4, 1.5e-4, 2e-4,  4e-4,  6e-4, 8e-4,  1e-3, 1.5e-3, 2e-3, 5e-3, 7e-3, 1e-2,2e-2, 5e-2, 8e-2, 1e-1])
    epsilon_base_list = []
    lambda_eps = np.ones([ len(epsilon_span), len(n_span)]) * np.nan

    list_legend = []
    fig, ax = plt.subplots(figsize=(8, 6))
    line_store =[]
    dots_store =[]

    for i, n in enumerate(n_span):
        mu, nv, x, y = util.sample_circle(n)
        epsilon_base = pointcloud.PointCloud(x, y).mean_cost_matrix
        epsilon_base_list.append(epsilon_base)
        for j, 𝜀_scale in enumerate(epsilon_span):
            epsilon = 𝜀_scale * epsilon_base
            threshold_n = 0.001 / (n**0.33)
            ot = SinkhornHessian.SinkhornHessian.solve_ott(x, y, mu, nv, epsilon, threshold_n)
            H = SinkhornHessian.SinkhornHessian.LHS_matrix(ot)
            S = jnp.linalg.svd(H, hermitian=True, compute_uv=False)
            lambda_eps[j,i]= S[-2]

        e = epsilon_span*np.array(epsilon_base_list[i])
        line_obj, = ax.plot(e, lambda_eps[:,i],marker='o',linestyle='--', markersize =7, linewidth=2, color=color_span[i],label = "N="+str(n))
        line_store.append(line_obj)
        dots_obj, = ax.plot(e, 4*np.pi**2/(n**3) * np.exp(-4*np.sin(np.pi/n)**2/(e)),color= color_span[i], linewidth=2, label ="$4\pi^2r_{"+str(n)+",\epsilon}/$"+str(n))
        dots_store.append(dots_obj)    

    
    plt.ylim([1e-17, 1e-3])
    plt.xlabel('$\epsilon$', fontsize =28)
    plt.ylabel('$\lambda_{2N-1}$', fontsize=25)
    plt.title('Equally spaced points on unit circle',fontsize =24)
    first_legend = ax.legend(handles=line_store, loc='upper left', ncol=1,fontsize=18)

    # Add the first legend manually to the current Axes
    ax.add_artist(first_legend)
    ax.legend(handles=dots_store, loc='lower right', fontsize=20)
    #plt.legend(fontsize=16)
    plt.yscale('log')
    plt.xscale('log') 
    plt.savefig("Unit_circle_lambda_eps.pdf", format="pdf", bbox_inches="tight")                             
    ###################################################


    ###################################################
    """ Case 2: 2D circle example: lambdas vs n """
    n_span = [50, 60, 70, 80, 90, 100, 120, 180, 240, 300, 400, 500, 800, 1000, 1200, 1500]
    epsilon_span  =[1e-5, 1e-4, 1e-3, 1e-2]

    epsilon_base_list = []
    lambda_eps = np.ones([ len(epsilon_span), len(n_span)]) * np.nan

    list_legend = []
    fig, ax = plt.subplots(figsize=(8, 6))
    line_store =[]
    dots_store =[]

    for i, n in enumerate(n_span):
        mu, nv, x, y = util.sample_circle(n)
        epsilon_base = pointcloud.PointCloud(x, y).mean_cost_matrix
        epsilon_base_list.append(epsilon_base)
        for j, 𝜀_scale in enumerate(epsilon_span):
            epsilon = 𝜀_scale * epsilon_base
            threshold_n = 0.001 / (n**0.33)
            ot = SinkhornHessian.SinkhornHessian.solve_ott(x, y, mu, nv, epsilon, threshold_n)
            H = SinkhornHessian.SinkhornHessian.LHS_matrix(ot)
            S = jnp.linalg.svd(H, hermitian=True, compute_uv=False)
            lambda_eps[j,i]= S[-2]

    for i, epsilon  in enumerate(epsilon_span):
        line_obj, =ax.plot(n_span, lambda_eps[i,:],marker='o',markersize= 7, linewidth=2, linestyle='--', color=color_span[i],label = "$\epsilon$="+str(epsilon))
        dots_obj, =ax.plot(n_span, np.array(epsilon_base_list)*epsilon/(4*np.array(n_span)), color= color_span[i], linewidth=2, label =str(epsilon)+'$/(4N)$')
        line_store.append(line_obj)
        dots_store.append(dots_obj)

        
    plt.ylim([1e-16, 2e-4])
    plt.xlabel('$N$', fontsize =28)
    plt.ylabel('$\lambda_{2N-1}$', fontsize=25)
    plt.title('Equally spaced points on unit circle',fontsize =24)
    first_legend = ax.legend(handles=line_store, loc='lower left', ncol=1,fontsize=20)

    # Add the first legend manually to the current Axes
    ax.add_artist(first_legend)
    ax.legend(handles=dots_store, loc='lower right', fontsize=20)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig("Unit_circle_lambda_N.pdf", format="pdf", bbox_inches="tight")
    ####################################################

    ####################################################
    """ Case 3: Uniform square example: lambda vs 1/epsilon """
    dim = 2
    n_span = [50,150, 500, 1500 ]
    epsilon_span = [ 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3,1.5e-3, 2e-3, 3e-3, 4e-3, 5e-3,7.5e-3,  1e-2,2e-2, 3e-2, 4e-2, 5e-2, 8e-2, 1e-1, 2e-1, 4e-1, 8e-1]
    epsilon_base_list = []
    lambda_eps = np.ones([ len(epsilon_span), len(n_span)]) * np.nan

    list_legend = []
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, n in enumerate(n_span):
        mu, nv, x, y = util.sample_points_uniform(n, n, dim, i)
        y=x
        epsilon_base = pointcloud.PointCloud(x, y).mean_cost_matrix
        epsilon_base_list.append(epsilon_base)

        for j, 𝜀_scale in enumerate(epsilon_span):
            epsilon =  𝜀_scale * epsilon_base
            threshold_n = 0.001 / (n**0.33)
            ot = SinkhornHessian.SinkhornHessian.solve_ott(x, y, mu, nv, epsilon, threshold_n)
            H = SinkhornHessian.SinkhornHessian.LHS_matrix(ot)
            S = jnp.linalg.svd(H, hermitian=True, compute_uv=False)
            lambda_eps[j,i]= S[-2]   

        e = epsilon_span*np.array(epsilon_base_list[i])
        plt.plot(1/e, lambda_eps[:,i],marker='o',markersize=10, linewidth=2, linestyle='--', color=color_span[i],label = "N="+str(n))


    plt.ylim([1e-17, 1e-4])
    plt.xlabel('$1/\epsilon$', fontsize =28)
    plt.ylabel('$\lambda_{2N-1}$', fontsize =25)
    plt.title('Uniformly distributed points in $[0,1]^2$',fontsize=24)
    plt.legend(loc = (0.25,0.0), fontsize=25)
    plt.yscale('log')
    plt.savefig("Uniform_lambda_1_eps.pdf", format="pdf", bbox_inches="tight")
    ####################################################

    ####################################################
    """ Case 4: Uniform square example: lambda vs n """
    dim    = 2
    n_span = [50, 70, 90, 100, 120, 180, 240, 300, 400, 500, 600,700, 800, 900, 1000, 1200, 1500,1700, 2000]
    epsilon_span  =np.array([  5e-3,1e-2, 2e-2, 4e-2])
    epsilon_base_list = []
    lambda_eps = np.ones([ len(epsilon_span), len(n_span)]) * np.nan

    list_legend = []
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, n in enumerate(n_span):
        mu, nv, x, y = util.sample_points_uniform(n, n, dim, i)
        y=x
        epsilon_base = pointcloud.PointCloud(x, y).mean_cost_matrix
        epsilon_base_list.append(epsilon_base)

        for j, 𝜀_scale in enumerate(epsilon_span):
            epsilon =  𝜀_scale * epsilon_base
            threshold_n = 0.001 / (n**0.33)
            ot = SinkhornHessian.SinkhornHessian.solve_ott(x, y, mu, nv, epsilon, threshold_n)
            H = SinkhornHessian.SinkhornHessian.LHS_matrix(ot)
            S = jnp.linalg.svd(H, hermitian=True, compute_uv=False)
            lambda_eps[j,i]= S[-2]  

    for j, epsilon  in enumerate(epsilon_span):
        plt.plot(n_span, lambda_eps[j,:],marker='o',markersize=10, linewidth=2, linestyle='--', color=color_span[j],label = "$\epsilon$="+str(epsilon))

    n_span_plot = np.linspace(500,2000,100)
    plt.plot(n_span_plot, 300*lambda_eps[3, 0]*n_span_plot**(-1), linewidth=4, label="$N^{-1}$")
    plt.ylim([5e-7, 1e-3])
    plt.xlabel('$N$', fontsize =28)
    plt.ylabel('$\lambda_{2N-1}$', fontsize=25)
    plt.title('Uniformly distributed points in $[0,1]^2$',fontsize=24)
    plt.legend(ncol=3, fontsize=19)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig("Uniform_lambda_N.pdf", format="pdf", bbox_inches="tight")
    ####################################################





