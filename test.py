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
n = 50000
m = 100000
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

#%%
geom = pointcloud.PointCloud(x, y, epsilon=epsilon, batch_size=16)
prob = linear_problem.LinearProblem(geom, a=mu, b=nv)
solver = sinkhorn.Sinkhorn(
            threshold=threshold, use_danskin=False, max_iterations=200000
            #solve_kwargs={
            #"implicit_diff": imp_diff.ImplicitDiff() if implicit else None}
        )
out = solver(prob)

f = out.f
g = out.g
a = out.geom.apply_transport_from_potentials(f, g, jnp.ones(m), axis=1)
b = out.geom.apply_transport_from_potentials(f, g, jnp.ones(n), axis=0)


svd_thr = 1e-10
SH = SinkhornHessian.SinkhornHessian(svd_thr)
H = SH.LHS_matrix(out)
T = SH.compute_hessian(out)
A = np.random.randn(n, dim)
result = jnp.tensordot(T, A, axes=((2,3), (0,1)))



def RA(A,x,y,f,g,a):
    vec1 = jnp.sum(x * A, axis=1)
    Mat1 = out.geom.apply_transport_from_potentials(f,g,y.T,axis=1)
    x1  = 2*(a * vec1 - jnp.sum(A * Mat1.T, axis=1))
    
    Mat2 = out.geom.apply_transport_from_potentials(f,g,A.T,axis=0)
    x2 = 2*(out.geom.apply_transport_from_potentials(f,g, vec1,axis=0) - jnp.sum(y * Mat2.T, axis=1))

    return x1, x2

def RAT(AT,xT,yT,f,g,a):
    vec1 = jnp.sum(xT * AT, axis=0)
    Mat1 = out.geom.apply_transport_from_potentials(f,g,yT,axis=1)
    x1  = 2*(a * vec1 - jnp.sum(AT * Mat1, axis=0))
    Mat2 = out.geom.apply_transport_from_potentials(f,g,AT,axis=0)
    x2 = 2*(out.geom.apply_transport_from_potentials(f,g, vec1,axis=0) - jnp.sum(yT * Mat2, axis=0))
    return x1, x2

def RTz(z1, z2, f, g, yT, a):
    vec1 = a * z1 
    Mat1 = x * vec1[:, None]
    Mat2 = out.geom.apply_transport_from_potentials(f,g,yT,axis=1) * z1
    vec2 = out.geom.apply_transport_from_potentials(f,g,z2,axis=1)
    Mat3 = x * vec2[:, None]
    Mat4 = out.geom.apply_transport_from_potentials(f, g, (yT * z2) , axis=1) 
    return 2*(Mat1 - Mat2.T + Mat3 - Mat4.T)



#%%

def solve_H_x(x1,x2, tau2, iter, epsilon, f, g, a, b):

    apply_potentials_1 = jax.jit(lambda x:  out.geom.apply_transport_from_potentials(f,g,x,axis=1))
    apply_potentials_0 = jax.jit(lambda x:  out.geom.apply_transport_from_potentials(f,g,x,axis=0))

    
    y1= x1/(a)
    y2 = -apply_potentials_0(y1) + x2
    m = len(g)

    #@jax.jit
    def T(z: Float[Array, str(m)]) -> Float[Array, str(m)]:
        piz = apply_potentials_1(z)
        piT_over_a_piz = apply_potentials_0(piz/a)
        return (b+epsilon*tau2)*z - piT_over_a_piz
        

    in_structure = jax.eval_shape(lambda: y2)
    fn_operator = lx.FunctionLinearOperator(T, in_structure, tags=lx.positive_semidefinite_tag)
    
    solver = lx.CG(rtol=1e-6, atol=1e-6, max_steps=iter)
    z = lx.linear_solve(fn_operator, y2, solver).value

    z1 = y1 - apply_potentials_1(z)/(a)
    z2 = z
    return z1, z2


# %%
def EA(A, epsilon, f, g, a):
    n =  A.shape[0]
    d =  A.shape[1]
    Mat1 = 2 * a[:, None] * A
    vec1 = jnp.sum(x * A, axis=1)
    Mat2 = -4/epsilon * x * (vec1*a)[:, None]
    Py   = out.geom.apply_transport_from_potentials(f,g,y.T,axis=1)
    PyT = Py.T
    Mat3 = 4/epsilon * PyT * vec1[:, None]
    vec2 = jnp.sum(PyT * A , axis=1)
    Mat4 = 4/epsilon * x * vec2[:, None]
    Mat5 = jnp.zeros((n,d))
    for i in range(d):
        YiY = y[:,i][:,None] * y
        Mat_i =  out.geom.apply_transport_from_potentials(f,g,YiY.T,axis=1).T
        vec_i = jnp.sum(Mat_i * A, axis=1)
        Mat5 = Mat5.at[:,i].set(-4/epsilon * vec_i)
 
    return Mat1 + Mat2 + Mat3 + Mat4 + Mat5

def HessianA(A,out):
    f = out.f
    g = out.g
    n = len(f)
    m = len(g)
    epsilon = out.geom.epsilon
    a = out.geom.apply_transport_from_potentials(f,g,jnp.ones(m),axis=1)
    b = out.geom.apply_transport_from_potentials(f,g,jnp.ones(n),axis=0)
    x = out.geom.x
    y = out.geom.y
    x1, x2 = RA(A,x,y,f,g,a)
    z1, z2 = solve_H_x(out,x1,x2, tau2, iter)
    return RTz(z1, z2)/epsilon+ EA(A, epsilon)


