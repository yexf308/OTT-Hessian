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


def _to_numpy(tensor: "torch.Tensor") -> np.ndarray:
    return np.array(tensor.detach().cpu().double().tolist())


jax.config.update("jax_enable_x64", True)


def _sample_problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, m, d = 4, 4, 2
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = rng.normal(size=(m, d)).astype(np.float32)
    mu = (np.ones(n) / n).astype(np.float32)
    nu = (np.ones(m) / m).astype(np.float32)
    return x, y, mu, nu


def _sinkhorn_jax(x, y, mu, nu, epsilon, threshold, max_iterations=1_000):
    del threshold  # tolerance-based stopping is not used in the JAX reference

    cost_matrix = 0.5 * jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    kernel = jnp.exp(-cost_matrix / epsilon)

    def _body(_, state):
        u, v = state
        Kv = kernel @ v
        u_new = mu / (Kv + 1e-12)
        KT_u = kernel.T @ u_new
        v_new = nu / (KT_u + 1e-12)
        return u_new, v_new

    init_state = (jnp.ones_like(mu), jnp.ones_like(nu))
    u_final, v_final = jax.lax.fori_loop(0, max_iterations, _body, init_state)

    plan = u_final[:, None] * kernel * v_final[None, :]
    transport_cost = jnp.sum(plan * cost_matrix)
    entropy_term = jnp.sum(plan * (jnp.log(plan + 1e-12) - 1.0))
    reg_cost = transport_cost + epsilon * entropy_term
    return reg_cost, plan


def _jax_cost(x, y, mu, nu, epsilon, threshold):
    cost, _ = _sinkhorn_jax(x, y, mu, nu, epsilon, threshold)
    return cost


def test_cost_and_hessian_match_jax_reference():
    x, y, mu, nu = _sample_problem(seed=5)
    epsilon = 0.05
    threshold = 1e-6

    x_jax = jnp.array(x)
    y_jax = jnp.array(y)
    mu_jax = jnp.array(mu)
    nu_jax = jnp.array(nu)

    hess_jax = np.array(jax.hessian(lambda pts: _jax_cost(pts, y_jax, mu_jax, nu_jax, epsilon, threshold))(x_jax))
    cost_jax = float(_jax_cost(x_jax, y_jax, mu_jax, nu_jax, epsilon, threshold))

    torch_backend = torch_sinkhorn_hessian.TorchSinkhornHessian(
        svd_thr=1e-12, max_iterations=5_000, use_compile=True
    )
    ot_torch = torch_backend.solve_ott(x, y, mu, nu, epsilon, threshold)
    hess_torch = _to_numpy(torch_backend.compute_hessian(ot_torch))

    # Transport costs should agree within a modest tolerance since both backends
    # solve the entropic OT problem.
    assert np.allclose(_to_numpy(ot_torch.reg_ot_cost), cost_jax, rtol=5e-2, atol=5e-2)

    assert np.allclose(hess_torch, hess_jax, rtol=1e-1, atol=1e-1)


def test_gradient_matches_jax_reference():
    x, y, mu, nu = _sample_problem(seed=7)
    epsilon = 0.05
    threshold = 1e-6

    x_jax = jnp.array(x)
    y_jax = jnp.array(y)
    mu_jax = jnp.array(mu)
    nu_jax = jnp.array(nu)

    grad_jax = np.array(jax.grad(lambda pts: _jax_cost(pts, y_jax, mu_jax, nu_jax, epsilon, threshold))(x_jax))

    torch_backend = torch_sinkhorn_hessian.TorchSinkhornHessian(
        svd_thr=1e-12, max_iterations=5_000, use_compile=True
    )
    ot_torch = torch_backend.solve_ott(x, y, mu, nu, epsilon, threshold)
    grad_torch = _to_numpy(torch_backend.dOTdx(ot_torch))

    assert np.allclose(grad_torch, grad_jax, rtol=5e-2, atol=5e-2)
