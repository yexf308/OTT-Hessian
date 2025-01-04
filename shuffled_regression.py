import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import time
import timeit
from jax.example_libraries import optimizers as jax_opt

import optax

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


if __name__ == '__main__':
    d_X        = 5
    d_Y        = 2
    n          = 500 
    noise      = 0.2 
    𝜀          = 0.05
    threshold  =  0.01 / (n**0.33)
    seed       = 2 
    blob_std   = [0.3, 0.05,0.6] 
    num_steps_sgd          = 15 
    sgd_learning_rate      = 0.001 
    n_s                    = 100
    num_steps_newton          = 50
    improvement_abs_threshold = 0.0001
    Patience                  = 3
    stop_counter              = 0 
    newton_learning_rate      = 0.25 
    abs_threshold             = 0.01  
    gd_learning_rate          = 0.001
    svd_thr                   = 1e-10
    num_steps_gd              = 2000

    mu, nv, y_jx,  x_jx, w_jx = util.sample_bolb(n, d_X, d_Y, blob_std, noise, seed)

    SR= SinkhornHessian.ShuffledRegression(x_jx, y_jx, mu, nv, 𝜀, threshold, num_steps_sgd, sgd_learning_rate, n_s, num_steps_newton, 
                 improvement_abs_threshold,Patience, newton_learning_rate, abs_threshold , gd_learning_rate, num_steps_gd, svd_thr)

    np.random.seed(1)
    w_int        = w_jx + 1*np.random.normal(size=[d_X,d_Y])
    w_int_jx     = jnp.array(w_int)

    method = "SGD-Newton"
    SGD_Newton_loss_list, SGD_Newton_grads_list, SGD_Newton_params_list = SR.fit( method, w_int_jx)

    method = "SGD-GD"
    SGD_GD_loss_list, SGD_GD_grads_list, SGD_GD_params_list = SR.fit( method, w_int_jx)

    np.savez('shuffled_regression.npz', SGD_Newton_loss_list=SGD_Newton_loss_list, SGD_Newton_grads_list=SGD_Newton_grads_list, SGD_Newton_params_list = SGD_Newton_params_list, SGD_GD_loss_list=SGD_GD_loss_list,
             SGD_GD_grads_list=SGD_GD_grads_list, SGD_GD_params_list = SGD_GD_params_list)
    
    # data = np.load('shuffled_regression.npz')
    # SGD_Newton_loss_list = data['SGD_Newton_loss_list']
    # SGD_Newton_grads_list = data['SGD_Newton_grads_list']
    # SGD_Newton_params_list = data['SGD_Newton_params_list']
    # SGD_GD_loss_list = data['SGD_GD_loss_list']
    # SGD_GD_grads_list = data['SGD_GD_grads_list']
    # SGD_GD_params_list = data['SGD_GD_params_list']

    # #############################################################################################################
    """1. SGD-GD plot: data points and the predicted points by the initial, 10, 500, and 2000 steps"""

    font = {'weight' : 'normal',
        'size'   : 16}
    matplotlib.rc('font', **font)
    markersize = 30

    fig, ax = plt.subplots(figsize=(6, 6))

    plt.rcParams.update({'font.size': 16})
    pred_jx = x_jx @ w_int_jx

    params = SGD_GD_params_list[10]
    pred_sgd_jx      = x_jx @ params


    params_gd =  SGD_GD_params_list[-1]
    pred_sgd_jx_final = x_jx @ params_gd

    params_gd =  SGD_GD_params_list[500]
    pred_sgd_jx_500 = x_jx @ params_gd


    ax.scatter(y_jx[:,0],y_jx[:,1],alpha=0.5, label=r"Target $Y^*$(noisy)")
    ax.scatter(pred_jx[:,0],pred_jx[:,1],alpha=0.5, marker = "x", 
            s=markersize, label=r"Initial $Y(\theta^{0})$")


    ax.scatter(pred_sgd_jx[:,0],pred_sgd_jx[:,1],alpha=0.5, marker = "*",  color = 'darkorchid',
            s=markersize, label="10 steps")

    ax.scatter(pred_sgd_jx_500[:,0],pred_sgd_jx_500[:,1],alpha=0.5, marker = "s", color = 'green',
            s=markersize, label="500 steps")

    ax.scatter(pred_sgd_jx_final[:,0],pred_sgd_jx_final[:,1],alpha=0.5, marker = "^", color = 'red',
            s=markersize, label="2000 steps")
    ax.legend(loc='best', fontsize=12)
    plt.xlabel('$y_1$')
    plt.ylabel('$y_2$')
    plt.savefig("shuffled_regression_gd.pdf", format="pdf", bbox_inches="tight")

    plt.show()

    # #############################################################################################################
    """2. SGD-GD plot: loss vs. iteration"""

    font = {'weight' : 'normal',
        'size'   : 18}

    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(figsize=(5, 5))

    plt.yscale('log')
    plt.xscale('log')
    plt.plot(np.arange(len(SGD_GD_loss_list)), SGD_GD_loss_list,lw=3)


    ax.set_xticks([10,100, 1000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks([1, 10, SGD_Newton_loss_list[-1],SGD_GD_loss_list[10],SGD_GD_loss_list[0]])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.xlabel('Iterations')
    plt.ylabel('Reg-OT Loss')
    plt.grid()
    plt.savefig("shuffled_regression_OT_loss_gd.pdf", format="pdf", bbox_inches="tight")

    plt.show()

    #############################################################################################################
    
    """3. SGD-GD plot: error vs. iteration"""
    error_gd = []

    for i in range(len(SGD_GD_params_list)):
        error_gd.append(jnp.linalg.norm(SGD_GD_params_list[i]-w_jx))
        
    fig, ax2 = plt.subplots(figsize=(5, 5))

    ax2.plot(np.arange(len(SGD_GD_params_list)), error_gd, label ='GD',lw=3)


    ax2.set_xticks([10,100, 1000])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_ylim([0.1,5])
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel(r'$\|\theta-\theta^*\|_2$')
    ax2.grid()
    plt.savefig("shuffled_regression_error_gd.pdf", format="pdf", bbox_inches="tight")

    plt.show()

    #############################################################################################################
    """4. OT_loss vs. iteration for SGD-Newton, SGD-GD"""

    font = {'weight' : 'normal',
            'size'   : 18}

    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(figsize=(5, 5))

    plt.yscale('log')
    plt.xscale('log')
    plt.plot(np.arange(11), SGD_Newton_loss_list[:11],label='First stage SGD',marker='o',lw=3,markersize=7)
    plt.plot(np.arange(10, len(SGD_Newton_loss_list)),SGD_Newton_loss_list[10:],lw=3,marker='o', markersize=7, label='Newton')
    plt.plot(np.arange(10, len(SGD_GD_loss_list)), SGD_GD_loss_list[10:], lw=3,marker='o',markersize=5,label ='GD')

    plt.axhline(y =SGD_Newton_loss_list[-1] , color = 'r', alpha=0.3,linestyle = '--') 
    plt.axhline(y =SGD_Newton_loss_list[10] , color = 'r', alpha=0.3,linestyle = '--') 

    ax.set_xticks([10,100, 1000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks([1, 10, SGD_Newton_loss_list[-1],SGD_GD_loss_list[1]])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.xlabel('Iterations')
    plt.ylabel('Reg-OT Loss')
    plt.grid()
    plt.legend(fontsize=14)
    plt.savefig("shuffled_regression_OT_loss.pdf", format="pdf", bbox_inches="tight")

    plt.show()

    #############################################################################################################
    
    """5. Error vs. iteration for SGD-Newton, SGD-GD"""
    error_newton = []
    error_gd = []
    for i in range(len(SGD_Newton_params_list)):
        error_newton.append(jnp.linalg.norm(SGD_Newton_params_list[i]-w_jx))
        
    for i in range(len(SGD_GD_params_list)):
        error_gd.append(jnp.linalg.norm(SGD_GD_params_list[i]-w_jx))
        
    fig, ax2 = plt.subplots(figsize=(5, 5))
    ax2.plot(np.arange(11), error_newton[:11],label='First stage SGD',marker='o',lw=3,markersize=7)
    ax2.plot(np.arange(10, len(SGD_Newton_params_list)), error_newton[10:], label='Newton',marker='o',lw=3,markersize=7)
    ax2.plot(np.arange(10, len(SGD_GD_params_list)), error_gd[10:], label ='GD',marker='o',lw=3,markersize=5)

    ax2.axhline(y =error_newton[-1] , color = 'r', alpha=0.3,linestyle = '--') 
    ax2.axhline(y =error_gd[10] , color = 'r', alpha=0.3,linestyle = '--') 
    ax2.axhline(y =error_gd[-1] , color = 'r', alpha=0.3,linestyle = '--') 

    ax2.set_xticks([10,100, 1000])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_yticks([error_newton[-1],error_gd[10],error_gd[-1]])
    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_ylim([0.1,5])
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel(r'$\|\theta-\theta^*\|_2$')
    ax2.grid()
    ax2.legend(fontsize=14)
    plt.savefig("shuffled_regression_error.pdf", format="pdf", bbox_inches="tight")

    plt.show()

