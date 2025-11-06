"""Detailed test to verify HVP implementation step by step."""

import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax
from jax import config
config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import SinkhornHessian

import torch
from torch_sinkhorn_hessian import TorchSinkhornHessian

# Test data
np.random.seed(42)
n, d = 8, 2
x = np.random.randn(n, d).astype(np.float64)
y = np.random.randn(n, d).astype(np.float64)
mu = np.ones(n) / n
nu = np.ones(n) / n
A = np.random.randn(n, d).astype(np.float64)
epsilon = 0.1

print("="*80)
print("Detailed HVP Verification")
print("="*80)
print(f"\nProblem size: n={n}, d={d}, epsilon={epsilon}")

# Solve OT in JAX
print("\n1. Solving OT with JAX/OTT...")
geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
prob = linear_problem.LinearProblem(geom, a=mu, b=nu)
solver_jax = sinkhorn.Sinkhorn(threshold=1e-6, max_iterations=10000)
out_jax = solver_jax(prob)
print(f"   Converged in {out_jax.n_iters} iterations")

# Solve OT in PyTorch
print("\n2. Solving OT with PyTorch...")
solver_torch = TorchSinkhornHessian(svd_thr=1e-10, dtype=torch.float64)
ot_torch = solver_torch.solve_ott(x, y, mu, nu, epsilon, 1e-6)
print(f"   Converged in {ot_torch.iterations} iterations")

# Compare transport plans
P_jax = np.array(out_jax.matrix)
P_torch = ot_torch.matrix.detach().cpu().numpy()
P_diff = np.max(np.abs(P_jax - P_torch))
print(f"\n3. Transport plan difference: {P_diff:.2e}")

# Method 1: Full Hessian in JAX
print("\n4. Computing full Hessian @ A in JAX...")
SH_jax = SinkhornHessian.SinkhornHessian(svd_thr=1e-10)
H_jax = SH_jax.compute_hessian(out_jax)
hvp_target_jax = jnp.tensordot(H_jax, A, axes=((2,3), (0,1)))
hvp_target = np.array(hvp_target_jax)
print(f"   Norm: {np.linalg.norm(hvp_target):.6f}")

# Method 2: Full Hessian in PyTorch
print("\n5. Computing full Hessian @ A in PyTorch...")
H_torch = solver_torch.compute_hessian(ot_torch)
hvp_from_torch_hess = torch.einsum('ijkl,kl->ij', H_torch, torch.tensor(A))
print(f"   Norm: {torch.norm(hvp_from_torch_hess).item():.6f}")

# Verify Hessians match
hess_diff = np.max(np.abs(np.array(H_jax) - H_torch.detach().cpu().numpy()))
print(f"\n6. Hessian difference: {hess_diff:.2e}")

# Method 3: HVP in PyTorch
print("\n7. Computing HVP in PyTorch...")
hvp_torch = solver_torch.hessian_vector_product(ot_torch, torch.tensor(A), tau2=1e-5, max_cg_iter=500)
print(f"   Norm: {torch.norm(hvp_torch).item():.6f}")

# Compare
diff_target = np.max(np.abs(hvp_target - hvp_torch.detach().cpu().numpy()))
diff_hess = torch.max(torch.abs(hvp_from_torch_hess - hvp_torch)).item()

print(f"\n8. Results:")
print(f"   HVP vs JAX target: {diff_target:.2e}")
print(f"   HVP vs PyTorch full Hessian: {diff_hess:.2e}")

if diff_hess < 0.1:
    print("\n✅ HVP matches PyTorch full Hessian!")
else:
    print(f"\n❌ Large difference: {diff_hess:.2e}")
    print("\nDebugging individual components...")
    
    # Test EA component alone
    apply_axis0, apply_axis1 = solver_torch._transport_functions(ot_torch)
    a_torch = apply_axis1(torch.ones(n, dtype=torch.float64))
    EA_torch = solver_torch._compute_EA(
        torch.tensor(A), epsilon, ot_torch.f, ot_torch.g, 
        a_torch, ot_torch.geom.x, ot_torch.geom.y, apply_axis1
    )
    print(f"   EA norm: {torch.norm(EA_torch).item():.6f}")

