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

from path import Path

if __name__ == '__main__':
    n1 = 500
    n2 = 1500
    n3 = 1500
    n = n1 + n2 + n3
    
    vector1 = [28,0,0]
    vector2 = [100,28,0]

    seed =10
    noise_level = 0.02

    d_X                         = 3
    d_Y                         = 3
    𝜀                           = 0.01
    threshold                   = 1e-3 
    num_steps_sgd               = 10 
    sgd_learning_rate           = 0.1 
    n_s                         = 500
    num_steps_newton            = 50
    improvement_abs_threshold   = 0.0001
    Patience                    = 3
    newton_learning_rate        = 0.5
    abs_threshold               = 0.001
    num_steps_gd                = 2000
    gd_learning_rate            = 0.05
    svd_thr                     = 1e-10

    mu           = np.ones((n,)) / n
    nv           = np.ones((n,)) / n

    point = util.room(n1, n2 ,n3, vector1, vector2)
    norm_pointcloud, noisy_rot_pointcloud_permute, matrix = util.noisy_rot_point(n, point, noise_level, seed)
    x_jx = jnp.array(norm_pointcloud)
    y_jx = jnp.array(noisy_rot_pointcloud_permute)
    w_jx = jnp.array(matrix)

    SR= SinkhornHessian.ShuffledRegression(x_jx, y_jx, mu, nv, 𝜀, threshold, num_steps_sgd, sgd_learning_rate, n_s, num_steps_newton,
                    improvement_abs_threshold,Patience, newton_learning_rate, abs_threshold , gd_learning_rate, num_steps_gd, svd_thr)
    seed = 2
    np.random.seed(seed)
    w_int_jx = jnp.array(matrix + 1*np.random.normal(size=[d_X,d_Y]))


    seed = 1
    np.random.seed(seed)
    method = "SGD-Newton"
    SGD_Newton_loss_list, SGD_Newton_grads_list, SGD_Newton_params_list = SR.fit( method, w_int_jx)

    method = "SGD-GD"
    SGD_GD_loss_list, SGD_GD_grads_list, SGD_GD_params_list = SR.fit( method, w_int_jx)
    
    np.savez('3D_pointcloud.npz', SGD_Newton_loss_list=SGD_Newton_loss_list, SGD_Newton_grads_list=SGD_Newton_grads_list, SGD_Newton_params_list = SGD_Newton_params_list, SGD_GD_loss_list=SGD_GD_loss_list,
                SGD_GD_grads_list=SGD_GD_grads_list, SGD_GD_params_list = SGD_GD_params_list)
    
    # data = np.load('3D_pointcloud.npz')
    # SGD_Newton_loss_list = data['SGD_Newton_loss_list']
    # SGD_Newton_grads_list = data['SGD_Newton_grads_list']
    # SGD_Newton_params_list = data['SGD_Newton_params_list']
    # SGD_GD_loss_list = data['SGD_GD_loss_list']
    # SGD_GD_grads_list = data['SGD_GD_grads_list']
    # SGD_GD_params_list = data['SGD_GD_params_list']


    #############################################################################################################



    """1. Plot 3D point cloud"""

    font = {'weight' : 'normal',
            'size'   : 15}

    matplotlib.rc('font', **font)
    fig = plt.figure()
    fig.set_size_inches(18, 10)

    spec = matplotlib.gridspec.GridSpec(ncols=6, nrows=2) # 6 columns evenly divides both 2 & 3

    ax1 = fig.add_subplot(spec[0,0:2],projection='3d') # row 0 with axes spanning 2 cols on evens
    ax2 = fig.add_subplot(spec[0,2:4],projection='3d')
    ax3 = fig.add_subplot(spec[0,4:],projection='3d')
    ax4 = fig.add_subplot(spec[1,0:2],projection='3d') # row 0 with axes spanning 2 cols on evens
    ax5 = fig.add_subplot(spec[1,2:4],projection='3d')
    ax6 = fig.add_subplot(spec[1,4:],projection='3d')



    ax1.scatter(norm_pointcloud[:,0], norm_pointcloud[:,1], norm_pointcloud[:,2], s=3)
    ax1.set_xlim([-0.7,0.7])
    ax1.set_ylim([-0.7,0.7])
    ax1.set_zlim([-0.5, 0.5])
    ax1.set_box_aspect([1.0, 1.0, 0.8])
    ax1.set_title("Original $X$")

    ax1.view_init(elev=40, azim=55, roll=0)


    ax2.scatter(noisy_rot_pointcloud_permute[:,0], noisy_rot_pointcloud_permute[:,1], noisy_rot_pointcloud_permute[:,2], s=3)
    ax2.set_xlim([-1,0.7])
    ax2.set_ylim([-1,1.5])
    ax2.set_zlim([-1, 1.5])
    ax2.set_box_aspect([1.0, 1.0, 0.8])
    ax2.set_title(r"Target $Y^*$(noisy)" )
    ax2.view_init(elev=40, azim=55, roll=0)



    y_target_sgd_int         = x_jx @ SGD_GD_params_list[0]
    y_target_sgd             = x_jx @ SGD_Newton_params_list[5]
    y_target_newton          = x_jx @ SGD_Newton_params_list[-1]
    y_target_gd              = x_jx @ SGD_GD_params_list[-1]


    ax3.scatter(y_target_sgd_int[:,0], y_target_sgd_int[:,1], y_target_sgd_int[:,2], s=3)
    ax3.set_xlim([-2,2])
    ax3.set_ylim([-2,2])
    ax3.set_zlim([-2, 2])
    ax3.set_box_aspect([1.0, 1.0, 0.8])
    ax3.set_title(r"Initial $Y(\theta^{0})$")
    ax3.view_init(elev=40, azim=55, roll=0)


    ax4.scatter(y_target_sgd[:,0], y_target_sgd[:,1], y_target_sgd[:,2], s=3)
    ax4.set_xlim([-2,2])
    ax4.set_ylim([-2,2])
    ax4.set_zlim([-2, 2])
    ax4.set_box_aspect([1.0, 1.0, 0.8])
    ax4.set_title(r"First stage SGD")
    ax4.view_init(elev=40, azim=55, roll=0)

    ax5.scatter(y_target_newton[:,0], y_target_newton[:,1], y_target_newton[:,2], s=3)
    ax5.set_xlim([-1,0.7])
    ax5.set_ylim([-1,1.5])
    ax5.set_zlim([-1, 1.5])
    ax5.set_box_aspect([1.0, 1.0, 0.8])
    ax5.set_title(r"Newton")
    ax5.view_init(elev=40, azim=55, roll=0)

    ax6.scatter(y_target_gd[:,0], y_target_gd[:,1], y_target_gd[:,2], s=3)
    ax6.set_xlim([-1,0.7])
    ax6.set_ylim([-1,1.5])
    ax6.set_zlim([-1, 1.5])
    ax6.set_box_aspect([1.0, 1.0, 0.8])
    ax6.set_title(r"GD")
    ax6.view_init(elev=40, azim=55, roll=0)

    plt.savefig("3d_pointcloud_result.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    #############################################################################################################
    
    """2. Plot loss vs. iteration for SGD-Newton, SGD-GD"""

    font = {'weight' : 'normal',
            'size'   : 15}

    matplotlib.rc('font', **font)

    fig = plt.figure()
    fig.set_size_inches(5, 5)
    ax1 = fig.add_subplot()
    plt.plot(np.arange(6), SGD_Newton_loss_list[:6],label='First stage SGD',marker='o',lw=3,markersize=7)
    plt.plot(np.arange(5, len(SGD_Newton_loss_list)),SGD_Newton_loss_list[5:] ,lw=3,marker='o', markersize=7, label='Newton')
    plt.plot(np.arange(5, len(SGD_GD_loss_list)), SGD_GD_loss_list[5:], lw=3,marker='o',markersize=5,label ='GD')

    plt.axhline(y =SGD_Newton_loss_list[-1] , color = 'r', alpha=0.3,linestyle = '--') 
    plt.axhline(y =SGD_Newton_loss_list[5] , color = 'r', alpha=0.3,linestyle = '--') 

    ax1.set_xticks([5, len(SGD_Newton_loss_list)])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.yticks([0.1, 0.5])
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Reg-OT Loss')
    ax1.set_title('')
    ax1.grid()
    ax1.legend(fontsize=14)
    plt.savefig("3d_pointcloud_OT_loss.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    #############################################################################################################
    
    """3. Error vs. iteration for SGD-Newton, SGD-GD"""

    error_newton = []
    error_gd = []
 
    for i in range(len(SGD_Newton_params_list)):
        error_newton.append(jnp.linalg.norm(SGD_Newton_params_list[i]-matrix))

    for i in range(len(SGD_GD_params_list)):
        error_gd.append(jnp.linalg.norm(SGD_GD_params_list[i]-matrix))
    fig = plt.figure()
    fig.set_size_inches(5, 5)    
    ax2 = fig.add_subplot()

    ax2.plot(np.arange(6), error_newton[:6],label='First stage SGD',marker='o',lw=3,markersize=7)
    ax2.plot(np.arange(5, len(SGD_Newton_params_list)), error_newton[5:],label='Newton',marker='o',lw=3,markersize=7)
    ax2.plot(np.arange(5, len(SGD_GD_params_list)), error_gd[5:], label ='GD',marker='o',lw=3,markersize=5)

    ax2.axhline(y =error_newton[-1] , color = 'r', alpha=0.3,linestyle = '--') 
    ax2.axhline(y =error_gd[10] , color = 'r', alpha=0.3,linestyle = '--') 
    ax2.axhline(y =error_gd[-1] , color = 'r', alpha=0.3,linestyle = '--') 

    ax2.set_yscale('log')
    ax2.set_xscale('log')

    ax2.set_xlabel('Iterations')
    ax2.set_ylabel(r'$\|\theta-\theta^*\|_2$')
    ax2.grid()
    ax2.legend(fontsize=14)
    plt.savefig("3d_pointcloud_error.pdf", format="pdf", bbox_inches="tight")
    plt.show()
