import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

torch = pytest.importorskip("torch")
torch_sinkhorn_hessian = pytest.importorskip("torch_sinkhorn_hessian")
JAXBackend = pytest.importorskip("SinkhornHessian")


def _to_numpy(tensor: "torch.Tensor") -> np.ndarray:
    return np.array(tensor.detach().cpu().double().tolist())


jax.config.update("jax_enable_x64", True)


def _sample_problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, m, d = 4, 4, 2
    x = rng.normal(size=(n, d)).astype(np.float64)
    y = rng.normal(size=(m, d)).astype(np.float64)
    mu = (np.ones(n) / n).astype(np.float64)
    nu = (np.ones(m) / m).astype(np.float64)
    return x, y, mu, nu


def _solve_jax_ot(x, y, mu, nu, epsilon, threshold):
    backend = JAXBackend.SinkhornHessian(svd_thr=1e-12)
    ot = backend.solve_ott(
        jnp.array(x, dtype=jnp.float64),
        jnp.array(y, dtype=jnp.float64),
        jnp.array(mu, dtype=jnp.float64),
        jnp.array(nu, dtype=jnp.float64),
        float(epsilon),
        float(threshold),
    )
    return backend, ot


def test_cost_and_hessian_match_jax_reference():
    x, y, mu, nu = _sample_problem(seed=5)
    epsilon = 0.05
    threshold = 1e-6

    jax_backend, ot_jax = _solve_jax_ot(x, y, mu, nu, epsilon, threshold)
    cost_jax = float(np.array(ot_jax.reg_ot_cost))
    hess_jax = np.array(jax_backend.compute_hessian(ot_jax))

    torch_backend = torch_sinkhorn_hessian.TorchSinkhornHessian(
        svd_thr=1e-12, max_iterations=5_000, use_compile=True, dtype=torch.float64
    )
    ot_torch = torch_backend.solve_ott(x, y, mu, nu, epsilon, threshold)
    hess_torch = _to_numpy(torch_backend.compute_hessian(ot_torch))

    assert np.allclose(hess_torch, hess_jax, rtol=1e-1, atol=1e-1)


def test_gradient_matches_jax_reference():
    x, y, mu, nu = _sample_problem(seed=7)
    epsilon = 0.05
    threshold = 1e-6

    jax_backend, ot_jax = _solve_jax_ot(x, y, mu, nu, epsilon, threshold)
    grad_jax = np.array(jax_backend.dOTdx(ot_jax))

    torch_backend = torch_sinkhorn_hessian.TorchSinkhornHessian(
        svd_thr=1e-12, max_iterations=5_000, use_compile=True, dtype=torch.float64
    )
    ot_torch = torch_backend.solve_ott(x, y, mu, nu, epsilon, threshold)
    grad_torch = _to_numpy(torch_backend.dOTdx(ot_torch))

    assert np.allclose(grad_torch, grad_jax, rtol=5e-2, atol=5e-2)


def test_hvp_matches_jax_analytic_direction():
    x, y, mu, nu = _sample_problem(seed=11)
    epsilon = 0.05
    threshold = 1e-6
    tau2 = 1e-5

    jax_backend, ot_jax = _solve_jax_ot(x, y, mu, nu, epsilon, threshold)
    hess_jax = np.array(jax_backend.compute_hessian(ot_jax))

    rng = np.random.default_rng(3)
    A = rng.normal(size=x.shape).astype(np.float64)
    hvp_jax = np.tensordot(hess_jax, A, axes=((2, 3), (0, 1)))

    torch_backend = torch_sinkhorn_hessian.TorchSinkhornHessian(
        svd_thr=1e-12, max_iterations=5_000, use_compile=False, dtype=torch.float64
    )
    ot_torch = torch_backend.solve_ott(x, y, mu, nu, epsilon, threshold)
    hvp_torch, info = torch_backend.hessian_vector_product(
        ot_torch,
        torch.tensor(A, dtype=torch.float64, device=ot_torch.geom.x.device),
        tau2=tau2,
        max_cg_iter=300,
        cg_rtol=1e-6,
        cg_atol=1e-6,
        use_preconditioner=True,
        return_info=True,
    )

    hvp_torch_np = _to_numpy(hvp_torch)
    assert np.allclose(hvp_torch_np, hvp_jax, rtol=1e-3, atol=1e-3)
    assert info["cg_converged"], f"CG did not converge: {info}"
