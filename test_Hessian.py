import jax
import jax.numpy as jnp
from jax import config
from jaxtyping import Array, Float 
config.update("jax_enable_x64", True)
import lineax as lx

import time
import timeit
from jax.example_libraries import optimizers as jax_opt


import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

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



#%%
n = 5000
m = 10000
epsilon = 0.01
dim = 5
threshold = 0.01 / (n**0.33)
tau2 =1e-5
iter = 100
mu, nv1, x, y1 = util.sample_points_uniform(n, n, dim, 1)
mu1, nv, x1, y = util.sample_points_uniform(m, m, dim, 1)
#y = x

xT = x.T
yT = y.T

#%% solve the optimal transport problem
geom = pointcloud.PointCloud(x, y, epsilon=epsilon, batch_size=16)
prob = linear_problem.LinearProblem(geom, a=mu, b=nv)
solver = sinkhorn.Sinkhorn(
            threshold=threshold, use_danskin=False, max_iterations=200000
            #solve_kwargs={
            #"implicit_diff": imp_diff.ImplicitDiff() if implicit else None}
        )
out = solver(prob)

#%% generate random matrix A

A = np.random.randn(n, dim)

#%% materialize the hessian
svd_thr = 1e-10
SH = SinkhornHessian.SinkhornHessian(svd_thr)
H = SH.LHS_matrix(out)
T = SH.compute_hessian(out)
result1 = jnp.tensordot(T, A, axes=((2,3), (0,1)))

#%% compute the hessian dot A without materializing the hessian
result2 = SinkhornHessian.HessianA(A,out, tau2, iter)

#%% compare the results
print(jnp.max(jnp.abs(result1 - result2)))

