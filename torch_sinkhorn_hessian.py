"""PyTorch implementation of the Sinkhorn Hessian utilities.

This module mirrors the interface of :mod:`SinkhornHessian` but relies on
PyTorch tensors and a simple Sinkhorn solver implemented directly in PyTorch.
It exposes convenience helpers that are compatible with the existing scripts
and allows differentiating the entropic OT objective with respect to the input
points through :mod:`torch.autograd`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import warnings

import torch

try:  # Optional import for GeomLoss-based solver
    from geomloss.samples_loss import sinkhorn_tensorized as _geomloss_sinkhorn
    from geomloss.sinkhorn_divergence import log_weights as _geomloss_log_weights
except Exception:  # pragma: no cover - GeomLoss is optional
    _geomloss_sinkhorn = None
    _geomloss_log_weights = None


# ---------------------------------------------------------------------------
# Helper structures


@dataclass
class _TorchGeometry:
    """Light-weight geometry container to mimic the JAX ott output."""

    x: torch.Tensor
    y: torch.Tensor
    epsilon: float


@dataclass
class TorchOTResult:
    """Container holding the solution of the balanced Sinkhorn problem."""

    geom: _TorchGeometry
    a: torch.Tensor
    b: torch.Tensor
    matrix: torch.Tensor
    reg_ot_cost: torch.Tensor
    threshold: float
    iterations: int
    f: Optional[torch.Tensor] = None  # Dual potentials
    g: Optional[torch.Tensor] = None


def _to_tensor(
    array,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Converts the provided ``array`` to a dense tensor on ``device``."""

    if isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=dtype)
    return torch.as_tensor(array, device=device, dtype=dtype)


def _compute_cost_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean ground cost used by the Sinkhorn solver.
    
    Note: Uses full squared Euclidean distance (without 0.5 factor) to match
    OTT's PointCloud default cost function (costs.SqEuclidean).
    """

    diff = x[:, None, :] - y[None, :, :]
    return torch.sum(diff * diff, dim=-1)


def _sinkhorn(
    x: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    epsilon: float,
    threshold: float,
    max_iterations: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Balanced Sinkhorn iterations with numerical stabilization.

    Uses log-domain Sinkhorn to avoid numerical underflow/overflow issues
    that occur with very small epsilon or large cost matrices.
    """

    cost_matrix = _compute_cost_matrix(x, y)
    
    # Log-domain Sinkhorn for numerical stability
    log_mu = torch.log(mu + 1e-300)
    log_nu = torch.log(nu + 1e-300)
    
    f = torch.zeros_like(mu)
    g = torch.zeros_like(nu)

    for iterations in range(1, max_iterations + 1):
        f_prev = f
        
        # Update f: f = epsilon * log(mu) - epsilon * log(sum_j exp((g_j - C_ij)/epsilon))
        # = epsilon * log(mu) - epsilon * logsumexp((g - C[i,:])/epsilon)
        temp = (g[None, :] - cost_matrix) / epsilon  # (n, m)
        f = epsilon * log_mu - epsilon * torch.logsumexp(temp, dim=1)
        
        # Update g: g = epsilon * log(nu) - epsilon * log(sum_i exp((f_i - C_ij)/epsilon))
        temp = (f[:, None] - cost_matrix) / epsilon  # (n, m)
        g = epsilon * log_nu - epsilon * torch.logsumexp(temp, dim=0)
        
        # Check convergence on f
        if torch.max(torch.abs(f - f_prev)).item() <= epsilon * threshold:
            break

    # Reconstruct transport plan from potentials
    transport_plan = torch.exp((f[:, None] + g[None, :] - cost_matrix) / epsilon)
    
    # Compute regularized cost
    transport_cost = torch.sum(transport_plan * cost_matrix)
    entropy_term = torch.sum(transport_plan * (torch.log(transport_plan + 1e-300) - 1.0))
    reg_cost = transport_cost + epsilon * entropy_term

    return reg_cost, transport_plan, cost_matrix, iterations, f, g


def _sinkhorn_compilable(
    x: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    epsilon: float,
    threshold: float,
    max_iterations: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sinkhorn iterations without Python side-effects for ``torch.compile``.

    The routine mirrors :func:`_sinkhorn` but avoids ``Tensor.item`` and early
    loop exits so that ``torch.compile`` can stage the computations. The number
    of iterations returned matches the first time the residual drops below the
    threshold when that happens, otherwise the maximum number of iterations is
    reported. The additional bookkeeping uses tensor operations exclusively so
    that the compiled graph remains valid.
    
    Uses log-domain Sinkhorn for numerical stability.
    """

    cost_matrix = _compute_cost_matrix(x, y)
    
    # Log-domain Sinkhorn for numerical stability
    log_mu = torch.log(mu + 1e-300)
    log_nu = torch.log(nu + 1e-300)
    
    f = torch.zeros_like(mu)
    g = torch.zeros_like(nu)

    threshold_tensor = torch.tensor(epsilon * threshold, dtype=x.dtype, device=x.device)
    iteration_ids = torch.arange(
        1, max_iterations + 1, dtype=torch.int64, device=x.device
    )
    recorded_iteration = torch.full((), max_iterations, dtype=torch.int64, device=x.device)
    has_converged = torch.zeros((), dtype=torch.bool, device=x.device)
    final_residual = torch.full((), float("inf"), dtype=x.dtype, device=x.device)

    for idx in range(max_iterations):
        f_prev = f
        
        # Update f
        temp = (g[None, :] - cost_matrix) / epsilon
        f = epsilon * log_mu - epsilon * torch.logsumexp(temp, dim=1)
        
        # Update g
        temp = (f[:, None] - cost_matrix) / epsilon
        g = epsilon * log_nu - epsilon * torch.logsumexp(temp, dim=0)

        diff = torch.max(torch.abs(f - f_prev))
        converged_now = diff <= threshold_tensor
        update_mask = torch.logical_and(~has_converged, converged_now)
        recorded_iteration = torch.where(update_mask, iteration_ids[idx], recorded_iteration)
        has_converged = torch.logical_or(has_converged, converged_now)
        final_residual = diff

    # Reconstruct transport plan from potentials
    transport_plan = torch.exp((f[:, None] + g[None, :] - cost_matrix) / epsilon)
    
    transport_cost = torch.sum(transport_plan * cost_matrix)
    entropy_term = torch.sum(transport_plan * (torch.log(transport_plan + 1e-300) - 1.0))
    reg_cost = transport_cost + epsilon * entropy_term

    return reg_cost, transport_plan, cost_matrix, recorded_iteration, final_residual, f, g


_COMPILED_SINKHORN = None
_COMPILED_MAX_ITERS = 10000  # Increased from 64 for better convergence


def _get_compiled_sinkhorn():
    """Return a compiled version of :func:`_sinkhorn` when available."""

    global _COMPILED_SINKHORN
    if _COMPILED_SINKHORN is not None:
        return _COMPILED_SINKHORN

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        _COMPILED_SINKHORN = _sinkhorn
        return _COMPILED_SINKHORN

    try:
        compiled_impl = compile_fn(
            _sinkhorn_compilable, fullgraph=True, backend="eager"
        )
    except Exception:  # pragma: no cover - compilation can legitimately fail
        _COMPILED_SINKHORN = _sinkhorn
        return _COMPILED_SINKHORN

    try:  # pragma: no cover - torch._dynamo may be unavailable on older builds
        from torch._dynamo.exc import InternalTorchDynamoError, Unsupported
    except Exception:  # pragma: no cover
        InternalTorchDynamoError = Unsupported = ()

    def _wrapped_sinkhorn(
        x: torch.Tensor,
        y: torch.Tensor,
        mu: torch.Tensor,
        nu: torch.Tensor,
        epsilon: float,
        threshold: float,
        max_iterations: int,
    ):
        effective_max = min(max_iterations, _COMPILED_MAX_ITERS)
        try:
            return compiled_impl(
                x,
                y,
                mu,
                nu,
                epsilon,
                threshold,
                effective_max,
            )
        except (Unsupported, InternalTorchDynamoError):
            cost, plan, cost_matrix, iterations = _sinkhorn(
                x,
                y,
                mu,
                nu,
                epsilon,
                threshold,
                max_iterations,
            )
            return (
                cost,
                plan,
                cost_matrix,
                torch.tensor(
                    iterations, device=cost.device, dtype=torch.int64
                ),
                torch.tensor(float("inf"), device=cost.device, dtype=cost.dtype),
            )

    _COMPILED_SINKHORN = _wrapped_sinkhorn
    return _COMPILED_SINKHORN


class TorchSinkhornHessian:
    """Sinkhorn Hessian computation class implemented with PyTorch."""

    def __init__(
        self,
        svd_thr: float,
        *,
        max_iterations: int = 1_000,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        use_compile: Optional[bool] = None,
        use_keops: bool = False,
        solver: str = "native",
        geomloss_scaling: float = 0.9,
    ) -> None:
        self.svd_thr = svd_thr
        self.max_iterations = max_iterations
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.use_keops = use_keops
        solver = solver.lower()
        if solver not in {"native", "geomloss"}:
            raise ValueError("solver must be either 'native' or 'geomloss'")
        self.solver = solver
        self.geomloss_scaling = float(geomloss_scaling)
        if self.solver == "geomloss":
            if _geomloss_sinkhorn is None or _geomloss_log_weights is None:
                raise ImportError(
                    "GeomLoss is not available. Install geomloss or use solver='native'."
                )
            self.use_compile = False
            sinkhorn_impl = None
        else:
            if use_compile is None:
                use_compile = getattr(torch, "compile", None) is not None
            if use_compile:
                sinkhorn_impl = _get_compiled_sinkhorn()
            else:
                sinkhorn_impl = _sinkhorn
            self.use_compile = sinkhorn_impl is not _sinkhorn
        self.last_cg_info: Optional[dict] = None
        self._sinkhorn = sinkhorn_impl

    # ------------------------------------------------------------------
    # Solver utilities

    def _prepare_inputs(
        self,
        x,
        y,
        mu,
        nu,
        epsilon: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        x_t = _to_tensor(x, device=self.device, dtype=self.dtype)
        y_t = _to_tensor(y, device=self.device, dtype=self.dtype)
        mu_t = _to_tensor(mu, device=self.device, dtype=self.dtype)
        nu_t = _to_tensor(nu, device=self.device, dtype=self.dtype)
        return x_t, y_t, mu_t, nu_t, float(epsilon)

    def _run_sinkhorn(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mu: torch.Tensor,
        nu: torch.Tensor,
        epsilon: float,
        threshold: float,
        max_iterations: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        if self.solver == "geomloss":
            return self._run_geomloss(
                x,
                y,
                mu,
                nu,
                epsilon,
            )
        result = self._sinkhorn(
            x,
            y,
            mu,
            nu,
            epsilon,
            threshold,
            max_iterations,
        )
        if len(result) == 7:
            cost, plan, cost_matrix, iterations, residual, f, g = result
        elif len(result) == 6:
            cost, plan, cost_matrix, iterations, f, g = result
            residual = None
        elif len(result) == 5:
            cost, plan, cost_matrix, iterations, residual = result
            # Approximate potentials from plan
            f = epsilon * torch.log(plan.sum(dim=1) + 1e-300)
            g = epsilon * torch.log(plan.sum(dim=0) + 1e-300)
        else:
            cost, plan, cost_matrix, iterations = result
            residual = None
            f = epsilon * torch.log(plan.sum(dim=1) + 1e-300)
            g = epsilon * torch.log(plan.sum(dim=0) + 1e-300)
            
        if isinstance(iterations, torch.Tensor):
            iterations_value = int(iterations.detach().cpu().item())
        else:
            iterations_value = int(iterations)
        return cost, plan, cost_matrix, iterations_value, residual, f, g

    def _run_geomloss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mu: torch.Tensor,
        nu: torch.Tensor,
        epsilon: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Sinkhorn solver backed by GeomLoss."""

        if _geomloss_sinkhorn is None or _geomloss_log_weights is None:
            raise RuntimeError("GeomLoss solver requested but geomloss is not installed.")

        # GeomLoss expects batched inputs; add batch dimension.
        x_b = x.unsqueeze(0)
        y_b = y.unsqueeze(0)
        mu_b = mu.unsqueeze(0)
        nu_b = nu.unsqueeze(0)

        # blur^2 corresponds to epsilon for p=2
        blur = torch.tensor(epsilon, dtype=self.dtype, device=self.device).sqrt()

        def _cost_fn(a_pts: torch.Tensor, b_pts: torch.Tensor) -> torch.Tensor:
            diff = a_pts[:, :, None, :] - b_pts[:, None, :, :]
            return torch.sum(diff * diff, dim=-1)

        f_ba, g_ab = _geomloss_sinkhorn(
            mu_b,
            x_b,
            nu_b,
            y_b,
            p=2,
            blur=float(blur.item()),
            scaling=self.geomloss_scaling,
            cost=_cost_fn,
            debias=False,
            potentials=True,
        )

        cost_matrix = _cost_fn(x_b, y_b).squeeze(0)

        epsilon_t = blur.pow(2)
        log_mu = torch.log(mu + 1e-300)
        log_nu = torch.log(nu + 1e-300)
        f = f_ba.squeeze(0) + epsilon_t * log_mu
        g = g_ab.squeeze(0) + epsilon_t * log_nu

        plan = torch.exp((f[:, None] + g[None, :] - cost_matrix) / epsilon_t)

        balance_tol = 1e-8
        balance_max_iter = 500
        for _ in range(balance_max_iter):
            row = plan.sum(dim=1)
            row_scaling = mu / (row + 1e-30)
            plan = plan * row_scaling[:, None]
            f = f + epsilon_t * torch.log(row_scaling)

            col = plan.sum(dim=0)
            col_scaling = nu / (col + 1e-30)
            plan = plan * col_scaling[None, :]
            g = g + epsilon_t * torch.log(col_scaling)

            if torch.max(torch.abs(row - mu)) < balance_tol and torch.max(torch.abs(col - nu)) < balance_tol:
                break

        reg_cost = torch.sum(plan * cost_matrix) + epsilon_t * torch.sum(plan * (torch.log(plan + 1e-30) - 1.0))
        iterations = -1  # GeomLoss hides internal iteration count
        return reg_cost, plan, cost_matrix, iterations, None, f, g

    def solve_ott(
        self,
        x,
        y,
        mu,
        nv,
        epsilon: float,
        threshold: float,
    ) -> TorchOTResult:
        x_t, y_t, mu_t, nv_t, eps = self._prepare_inputs(x, y, mu, nv, epsilon)
        cost, plan, _, iterations_value, residual, f, g = self._run_sinkhorn(
            x_t,
            y_t,
            mu_t,
            nv_t,
            eps,
            threshold,
            self.max_iterations,
        )
        if (
            residual is not None
            and residual.detach().cpu().item() > threshold
            and self.max_iterations > _COMPILED_MAX_ITERS
        ):
            cost, plan, _, iterations_value, _, f, g = self._run_sinkhorn(
                x_t,
                y_t,
                mu_t,
                nv_t,
                eps,
                threshold,
                self.max_iterations,
            )
        geom = _TorchGeometry(x=x_t, y=y_t, epsilon=eps)
        return TorchOTResult(
            geom=geom,
            a=mu_t,
            b=nv_t,
            matrix=plan,
            reg_ot_cost=cost,
            threshold=threshold,
            iterations=iterations_value,
            f=f,
            g=g,
        )

    def solve_ott_cost(
        self,
        x,
        y,
        mu,
        nv,
        epsilon: float,
        threshold: float,
    ) -> torch.Tensor:
        return self.solve_ott(x, y, mu, nv, epsilon, threshold).reg_ot_cost

    def solve_ott_implicit_cost(
        self,
        x,
        y,
        mu,
        nv,
        epsilon: float,
        threshold: float,
    ) -> torch.Tensor:
        # The explicit solver is fully differentiable; implicit differentiation
        # is not required for the PyTorch backend.
        return self.solve_ott_cost(x, y, mu, nv, epsilon, threshold)

    # ------------------------------------------------------------------
    # Helper methods for analytical Hessian computation

    @staticmethod
    def _LHS_matrix(ot: TorchOTResult) -> torch.Tensor:
        """Construct the left-hand side matrix for Hessian computation."""
        a = ot.a
        b = ot.b
        P = ot.matrix
        a_P = torch.sum(P, dim=1)
        b_P = torch.sum(P, dim=0)

        a_diag = torch.diag(a_P)
        b_diag = torch.diag(b_P)
        PT = P.t()

        H1 = torch.cat([a_diag, P], dim=1)
        H2 = torch.cat([PT, b_diag], dim=1)
        H = torch.cat([H1, H2], dim=0)

        return H

    @staticmethod
    def _RHS(ot: TorchOTResult) -> torch.Tensor:
        """Construct the right-hand side tensor for Hessian computation."""
        x = ot.geom.x
        y = ot.geom.y
        P = ot.matrix
        
        # Cost gradient: dC/dx = 2(x - y) for squared Euclidean distance
        dCk_dxk = 2 * (x[:, None, :] - y[None, :, :])
        
        # b_g: gradient contribution from target points
        b_g = dCk_dxk * P[:, :, None]  # (M, N, D)
        b_g = b_g.transpose(0, 1)  # (N, M, D)
        b_g_col = torch.sum(b_g, dim=0)  # (M, D)

        M, D = b_g_col.shape
        # b_f: diagonal gradient contribution
        b_f = torch.zeros((M, M, D), device=x.device, dtype=x.dtype)
        indices = torch.arange(M, device=x.device)
        b_f[indices, indices, :] = b_g_col

        b = torch.cat([b_f, b_g], dim=0)
        return b

    # ------------------------------------------------------------------
    # Differential quantities

    def dOTdx(self, ot: TorchOTResult) -> torch.Tensor:
        if self.solver == "geomloss":
            x = ot.geom.x
            y = ot.geom.y
            plan = ot.matrix
            # Gradient of 0.5‖x-y‖^2? For full squared cost we use 2*(x-y)
            return torch.sum(plan[:, :, None] * 2.0 * (x[:, None, :] - y[None, :, :]), dim=1)

        x = ot.geom.x.clone().detach().requires_grad_(True)
        cost, _, _, _, _, _, _ = self._run_sinkhorn(
            x,
            ot.geom.y,
            ot.a,
            ot.b,
            ot.geom.epsilon,
            ot.threshold,
            self.max_iterations,
        )
        (grad,) = torch.autograd.grad(cost, x)
        return grad

    def compute_hessian_no_reg(self, ot: TorchOTResult) -> torch.Tensor:
        """Compute analytical Hessian without regularization."""
        epsilon = ot.geom.epsilon
        H = self._LHS_matrix(ot)
        nm = H.shape[0]
        R = self._RHS(ot)
        m = R.shape[1]
        dim = R.shape[2]

        # Solve H @ HdagR = R
        R_reshape = R.reshape(nm, m * dim)
        HdagR_reshape = torch.linalg.solve(H, R_reshape)
        HdagR = HdagR_reshape.reshape(nm, m, dim)
        
        # First part of Hessian: R^T @ H^{-1} @ R / epsilon
        Hessian_1 = torch.einsum('skd,sjt->kdjt', R, HdagR) / epsilon

        # Second part: direct cost Hessian
        x = ot.geom.x
        y = ot.geom.y
        P = ot.matrix
        dCk_dxk = 2 * (x[:, None, :] - y[None, :, :])
        d2Ck_dx2k = 2
        M, N, D = dCk_dxk.shape

        weighted_C = -dCk_dxk / epsilon * P[:, :, None]
        Hessian_2_part = torch.einsum('kjs,kjt->kst', weighted_C, dCk_dxk)
        Hessian_3_diag = torch.sum(d2Ck_dx2k * P, dim=1)

        identity_matrix = torch.eye(D, device=x.device, dtype=x.dtype)
        expanded_Hessian_3_diag = Hessian_3_diag[:, None, None]
        G = Hessian_2_part + expanded_Hessian_3_diag * identity_matrix

        Hessian_2 = torch.zeros((M, D, M, D), device=x.device, dtype=x.dtype)
        indices = torch.arange(M, device=x.device)
        Hessian_2[indices, :, indices, :] = G

        Hessian = Hessian_1 + Hessian_2
        return Hessian

    def compute_hessian(self, ot: TorchOTResult) -> torch.Tensor:
        """Compute analytical Hessian with SVD regularization."""
        epsilon = ot.geom.epsilon
        H = self._LHS_matrix(ot)
        R = self._RHS(ot)

        # Apply SVD regularization
        eigenvalues, eigenvectors = torch.linalg.eigh(H)
        eigenvalues_sqrt_inv = torch.where(
            eigenvalues > self.svd_thr,
            1 / torch.sqrt(eigenvalues),
            torch.zeros_like(eigenvalues)
        )
        Hsqrt = eigenvectors * eigenvalues_sqrt_inv[None, :]
        bHsqrt = torch.einsum('ikd,is->ksd', R, Hsqrt)
        Hessian_1 = torch.einsum('ksd,jst->kdjt', bHsqrt, bHsqrt) / epsilon

        # Second part: direct cost Hessian
        x = ot.geom.x
        y = ot.geom.y
        P = ot.matrix
        dCk_dxk = 2 * (x[:, None, :] - y[None, :, :])
        d2Ck_dx2k = 2
        M, N, D = dCk_dxk.shape

        weighted_C = -dCk_dxk / epsilon * P[:, :, None]
        Hessian_2_part = torch.einsum('kjs,kjt->kst', weighted_C, dCk_dxk)
        Hessian_3_diag = torch.sum(d2Ck_dx2k * P, dim=1)

        identity_matrix = torch.eye(D, device=x.device, dtype=x.dtype)
        expanded_Hessian_3_diag = Hessian_3_diag[:, None, None]
        G = Hessian_2_part + expanded_Hessian_3_diag * identity_matrix

        Hessian_2 = torch.zeros((M, D, M, D), device=x.device, dtype=x.dtype)
        indices = torch.arange(M, device=x.device)
        Hessian_2[indices, :, indices, :] = G

        Hessian = Hessian_1 + Hessian_2
        return Hessian

    def hess_loss_implicit(self, x, y, mu, nv, epsilon: float, threshold: float) -> torch.Tensor:
        """Compute Hessian using analytical formula (same as analytical for PyTorch)."""
        # In PyTorch, we use the analytical formula for all cases
        # since implicit differentiation would require additional infrastructure
        return self.hess_loss_analytical(x, y, mu, nv, epsilon, threshold)

    def hess_loss_unroll(self, x, y, mu, nv, epsilon: float, threshold: float) -> torch.Tensor:
        """Compute Hessian using analytical formula (same as analytical for PyTorch)."""
        # In PyTorch, we use the analytical formula for all cases
        return self.hess_loss_analytical(x, y, mu, nv, epsilon, threshold)

    def hess_loss_analytical(self, x, y, mu, nv, epsilon: float, threshold: float) -> torch.Tensor:
        """Compute Hessian using analytical formula with SVD regularization."""
        ot = self.solve_ott(x, y, mu, nv, epsilon, threshold)
        return self.compute_hessian(ot)

    def hess_loss_analytical_no_reg(
        self,
        x,
        y,
        mu,
        nv,
        epsilon: float,
        threshold: float,
    ) -> torch.Tensor:
        """Compute Hessian using analytical formula without regularization."""
        ot = self.solve_ott(x, y, mu, nv, epsilon, threshold)
        return self.compute_hessian_no_reg(ot)

    # ------------------------------------------------------------------
    # Hessian-Vector Products (Memory Efficient for Large-Scale Problems)

    def _transport_functions(self, ot: TorchOTResult):
        """Create transport application functions from OT potentials.
        
        These apply the transport plan P to vectors without materializing P.
        Uses KeOps for memory efficiency when enabled.
        """
        # Use actual potentials from OT solution
        f = ot.f if ot.f is not None else ot.geom.epsilon * torch.log(ot.matrix.sum(dim=1) + 1e-300)
        g = ot.g if ot.g is not None else ot.geom.epsilon * torch.log(ot.matrix.sum(dim=0) + 1e-300)
        epsilon = ot.geom.epsilon
        
        geom = ot.geom
        P = ot.matrix
        
        if self.use_keops:
            # Use KeOps for memory-efficient lazy evaluation
            try:
                from pykeops.torch import LazyTensor
                
                x_i = LazyTensor(geom.x[:, None, :])
                y_j = LazyTensor(geom.y[None, :, :])
                f_i = LazyTensor(f[:, None, None])
                g_j = LazyTensor(g[None, :, None])
                
                # Cost matrix (lazy)
                C_ij = ((x_i - y_j) ** 2).sum(dim=2)
                
                # Kernel K_ij = exp((f_i + g_j - C_ij) / epsilon)
                K_ij = ((f_i + g_j - C_ij) / epsilon).exp()
                
                def apply_axis0(arr):
                    """Apply P^T @ arr (transport from target to source)."""
                    if arr.ndim == 1:
                        arr = arr[None, :]
                        squeeze_output = True
                    else:
                        squeeze_output = False
                    
                    # arr is (batch, m), result is (batch, n)
                    arr_j = LazyTensor(arr[:, None, :])  # (batch, 1, m)
                    result = (K_ij * arr_j).sum(dim=1)  # Sum over j, result (batch, n)
                    
                    if squeeze_output:
                        result = result[0]
                    return result
                
                def apply_axis1(arr):
                    """Apply P @ arr (transport from source to target)."""
                    if arr.ndim == 1:
                        arr = arr[None, :]
                        squeeze_output = True
                    else:
                        squeeze_output = False
                    
                    # arr is (batch, m), result is (batch, n)
                    arr_i = LazyTensor(arr[:, None, :])  # (batch, 1, m)
                    result = (K_ij * arr_i).sum(dim=0)  # Sum over i? Need to check
                    
                    if squeeze_output:
                        result = result[0]
                    return result
                    
            except ImportError:
                # Fallback to standard PyTorch if KeOps not available
                self.use_keops = False
                return self._transport_functions(ot)
        else:
            # Standard PyTorch implementation using potentials (matches OTT's LSE approach)
            # This is more numerically stable than using materialized P
            cost_matrix = torch.sum(
                (geom.x[:, None, :] - geom.y[None, :, :]) ** 2, dim=-1
            )
            
            def _apply_lse(f_pot, g_pot, vec, axis_val):
                """Apply transport using log-sum-exp (numerically stable like OTT).
                
                Computes (P @ vec) or (P.T @ vec) where P = exp((f + g - C) / eps)
                """
                if vec.ndim == 1:
                    vec = vec[None, :]  # Add batch dimension
                    squeeze_output = True
                else:
                    squeeze_output = False
                
                # vec can have positive and negative values, so we can't take log directly
                # Instead, use the LSE formula with signs
                # result = sum_k exp(...) * vec[k] = sum_k exp(... + log|vec[k]|) * sign(vec[k])
                
                # For simplicity with mixed signs, use direct exponentiation
                # (This is what OTT does when vec has mixed signs)
                kernel = torch.exp((f_pot[:, None] + g_pot[None, :] - cost_matrix) / epsilon)
                
                if axis_val == 0:
                    # P.T @ vec where vec is (batch, n)
                    # result is (batch, m)
                    result = torch.matmul(vec, kernel)  # (batch, n) @ (n, m) = (batch, m)
                else:  # axis == 1
                    # P @ vec where vec is (batch, m)
                    # result is (batch, n)
                    result = torch.matmul(vec, kernel.t())  # (batch, m) @ (m, n) = (batch, n)
                
                if squeeze_output:
                    result = result[0]
                
                return result
            
            def apply_axis0(arr):
                """Apply P.T @ arr using potentials."""
                return _apply_lse(f, g, arr, axis_val=0)
            
            def apply_axis1(arr):
                """Apply P @ arr using potentials."""
                return _apply_lse(f, g, arr, axis_val=1)
        
        return apply_axis0, apply_axis1

    def _compute_RA(self, A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, 
                    apply_axis0, apply_axis1):
        """Compute R @ A terms. Ported from test.py RA function."""
        # vec1[i] = sum_d x[i,d] * A[i,d]
        vec1 = torch.sum(x * A, dim=1)  # (n,)
        
        # Mat1 = apply_axis1(y.T): (d, m) → (d, n)
        Mat1 = apply_axis1(y.t())  # (d, n)
        
        # x1[i] = 2 * (a[i] * vec1[i] - sum_d A[i,d] * Mat1[d,i])
        # Note: a is computed inside hessian_vector_product
        # Return without multiplying by 'a' yet
        x1_no_a = vec1 - torch.sum(A * Mat1.t(), dim=1)  # (n,)
        
        # Mat2 = apply_axis0(A.T): (d, n) → (d, m)
        Mat2 = apply_axis0(A.t())  # (d, m)
        
        # x2[j] = 2 * (apply_axis0(vec1)[j] - sum_d y[j,d] * Mat2[d,j])
        x2 = 2 * (apply_axis0(vec1) - torch.sum(y * Mat2.t(), dim=1))  # (m,)
        
        return x1_no_a, x2

    def _compute_EA(self, A: torch.Tensor, epsilon: float, f: torch.Tensor, g: torch.Tensor,
                    a: torch.Tensor, x: torch.Tensor, y: torch.Tensor, apply_axis1):
        """Compute E @ A terms. Based on test.py implementation."""
        n, d = A.shape
        
        # Mat1 = 2 * a * A (element-wise, with broadcasting)
        Mat1 = 2 * a[:, None] * A  # (n, d)
        
        # vec1[i] = sum_d x[i,d] * A[i,d]
        vec1 = torch.sum(x * A, dim=1)  # (n,)
        
        # Mat2 = -4/epsilon * x * (vec1 * a)
        Mat2 = -4 / epsilon * x * (vec1 * a)[:, None]  # (n, d)
        
        # Py = apply_axis1(y.T): (d, m) → (d, n)
        Py = apply_axis1(y.t())  # (d, n)
        PyT = Py.t()  # (n, d)
        
        # Mat3 = 4/epsilon * PyT * vec1
        Mat3 = 4 / epsilon * PyT * vec1[:, None]  # (n, d)
        
        # vec2[i] = sum_d PyT[i,d] * A[i,d]
        vec2 = torch.sum(PyT * A, dim=1)  # (n,)
        
        # Mat4 = 4/epsilon * x * vec2
        Mat4 = 4 / epsilon * x * vec2[:, None]  # (n, d)
        
        # Mat5: for each dimension
        Mat5 = torch.zeros_like(A)  # (n, d)
        for i in range(d):
            YiY = y[:, i:i+1] * y  # (m, d)
            Mat_i = apply_axis1(YiY.t()).t()  # (d, m) → (d, n) → (n, d)
            vec_i = torch.sum(Mat_i * A, dim=1)  # (n,)
            Mat5[:, i] = -4 / epsilon * vec_i
        
        return Mat1 + Mat2 + Mat3 + Mat4 + Mat5

    def _compute_RTz(self, z1: torch.Tensor, z2: torch.Tensor,
                     x: torch.Tensor, y: torch.Tensor, a: torch.Tensor,
                     apply_axis1):
        """Compute R^T @ z terms. Based on test.py implementation."""
        yT = y.t()  # (d, m)
        
        vec1 = a * z1  # (n,)
        Mat1 = x * vec1[:, None]  # (n, d)
        
        # Mat2 = apply_axis1(yT) * z1: (d, n) * (n,) with broadcast
        Mat2_raw = apply_axis1(yT)  # (d, m) → (d, n)
        Mat2 = Mat2_raw * z1[None, :]  # (d, n) * (1, n) = (d, n)
        
        vec2 = apply_axis1(z2)  # (m,) → (n,)
        Mat3 = x * vec2[:, None]  # (n, d)
        
        # Mat4 = apply_axis1(yT * z2): (d, m) → (d, n)
        Mat4 = apply_axis1(yT * z2[None, :])  # (d, n)
        
        return 2 * (Mat1 - Mat2.t() + Mat3 - Mat4.t())  # (n, d)


    def hessian_vector_product(
        self,
        ot: TorchOTResult,
        A,
        tau2: float = 1e-5,
        max_cg_iter: int = 300,
        *,
        cg_rtol: float = 1e-6,
        cg_atol: float = 1e-6,
        use_preconditioner: bool = True,
        return_info: bool = False,
    ):
        """Compute Hessian @ A without materializing the Hessian.
        
        Memory-efficient for large-scale problems. Ported from test.py.
        
        Args:
            ot: OT solution containing transport plan and potentials.
            A: Vector/matrix to multiply with Hessian (shape: n × d).
            tau2: Regularization parameter for linear system.
            max_cg_iter: Maximum iterations for conjugate gradient solver.
            cg_rtol: Relative tolerance for CG termination.
            cg_atol: Absolute tolerance for CG termination.
            use_preconditioner: If True, apply Neumann-style preconditioning mirroring the
                JAX implementation for faster convergence.
            return_info: When True, also return CG diagnostics (`dict`).
            
        Returns:
            Hessian @ A (shape: n × d) without ever materializing full Hessian. When
            ``return_info`` is True, also returns a diagnostics dictionary with CG stats.
            
        Memory: O(n*d) instead of O(n²*d²) for materialized Hessian
        """
        f = ot.f
        g = ot.g
        x = ot.geom.x  # (n, d)
        y = ot.geom.y  # (m, d)
        epsilon = ot.geom.epsilon
        n = x.shape[0]
        m = y.shape[0]

        # Ensure direction is a tensor on the correct device/dtype
        A = _to_tensor(A, device=x.device, dtype=x.dtype)
        
        # Get transport functions
        apply_axis0, apply_axis1 = self._transport_functions(ot)
        
        # Compute marginals
        ones_m = torch.ones(m, device=x.device, dtype=x.dtype)
        ones_n = torch.ones(n, device=x.device, dtype=x.dtype)
        a = apply_axis1(ones_m)  # (n,)
        b = apply_axis0(ones_n)  # (m,)
        
        # Step 1: Compute R @ A (following test.py exactly)
        vec1 = torch.sum(x * A, dim=1)  # (n,)
        Mat1 = apply_axis1(y.t())  # (d, m) → (d, n)
        x1 = 2 * (a * vec1 - torch.sum(A * Mat1.t(), dim=1))  # (n,)
        
        Mat2 = apply_axis0(A.t())  # (d, n) → (d, m)
        x2 = 2 * (apply_axis0(vec1) - torch.sum(y * Mat2.t(), dim=1))  # (m,)
        
        # Step 2: Solve linear system
        y1 = x1 / a  # (n,)
        y2_raw = -apply_axis0(y1) + x2  # (m,)
        denom = b + epsilon * tau2

        def _apply_B(z_vec: torch.Tensor) -> torch.Tensor:
            z_vec = z_vec.reshape(-1)
            piz = apply_axis1(z_vec)
            piT_over_a_piz = apply_axis0(piz / a)
            return piT_over_a_piz / denom

        if use_preconditioner:
            rhs = y2_raw / denom

            def linear_op(z_vec: torch.Tensor) -> torch.Tensor:
                return z_vec - _apply_B(z_vec)

            def preconditioner(vec: torch.Tensor) -> torch.Tensor:
                b1 = _apply_B(vec)
                b2 = _apply_B(b1)
                b3 = _apply_B(b2)
                return vec + b1 + b2 + b3
        else:
            rhs = y2_raw

            def linear_op(z_vec: torch.Tensor) -> torch.Tensor:
                z_vec = z_vec.reshape(-1)
                piz = apply_axis1(z_vec)
                piT_over_a_piz = apply_axis0(piz / a)
                return denom * z_vec - piT_over_a_piz

            preconditioner = None  # type: ignore

        # Solve using CG
        z, cg_info = self._conjugate_gradient(
            linear_op,
            rhs,
            max_iter=max_cg_iter,
            rtol=cg_rtol,
            atol=cg_atol,
            preconditioner=preconditioner if use_preconditioner else None,
            return_info=True,
        )
        self.last_cg_info = cg_info
        if not cg_info["cg_converged"]:
            warnings.warn(
                (
                    "Conjugate gradient did not converge within the allotted iterations. "
                    f"residual={cg_info['cg_residual']:.3e} "
                    f"(initial={cg_info['cg_initial_residual']:.3e}, "
                    f"iters={cg_info['cg_iters']})."
                ),
                RuntimeWarning,
            )
        
        # Step 3: Compute z1, z2
        z1 = y1 - apply_axis1(z) / a  # (n,)
        z2 = z  # (m,)
        
        # Step 4: Compute RTz (R^T @ z)
        yT = y.t()  # (d, m)
        vec1_z = a * z1  # (n,)
        Mat1_z = x * vec1_z[:, None]  # (n, d)
        
        Mat2_raw = apply_axis1(yT)  # (d, m) → (d, n)
        Mat2 = Mat2_raw * z1[None, :]  # (d, n)
        
        vec2_z = apply_axis1(z2)  # (m,) → (n,)
        Mat3_z = x * vec2_z[:, None]  # (n, d)
        
        Mat4 = apply_axis1(yT * z2[None, :])  # (d, n)
        
        RTz = 2 * (Mat1_z - Mat2.t() + Mat3_z - Mat4.t())  # (n, d)
        
        # Step 5: Compute EA (E @ A)
        EA = self._compute_EA(A, epsilon, f, g, a, x, y, apply_axis1)  # (n, d)
        
        # Step 6: Combine
        hvp = RTz / epsilon + EA
        if return_info:
            return hvp, cg_info
        return hvp

    def _conjugate_gradient(
        self,
        matvec,
        b: torch.Tensor,
        max_iter: int = 100,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        preconditioner=None,
        return_info: bool = False,
    ):
        """Simple conjugate gradient solver for Ax = b.
        
        Args:
            matvec: Function that computes A @ x
            b: Right-hand side
            max_iter: Maximum iterations
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            Solution x (and optional diagnostic dictionary when ``return_info`` is True).
        """
        x = torch.zeros_like(b)
        r = b.clone()
        b_norm = torch.linalg.norm(b)

        if b_norm == 0:
            info = {
                "cg_residual": 0.0,
                "cg_iters": 0,
                "cg_converged": True,
                "cg_initial_residual": 0.0,
            }
            return (x, info) if return_info else x

        if preconditioner is not None:
            z = preconditioner(r)
        else:
            z = r.clone()

        p = z.clone()
        rz_old = torch.dot(r, z)
        residual_norm = torch.linalg.norm(r)
        converged = residual_norm <= atol + rtol * b_norm
        num_iters = 0

        for i in range(max_iter):
            if converged:
                num_iters = i
                break

            Ap = matvec(p)
            denom = torch.dot(p, Ap) + 1e-12
            alpha = rz_old / denom
            x = x + alpha * p
            r = r - alpha * Ap
            residual_norm = torch.linalg.norm(r)
            converged = residual_norm <= atol + rtol * b_norm
            if converged:
                num_iters = i + 1
                break

            if preconditioner is not None:
                z = preconditioner(r)
            else:
                z = r

            rz_new = torch.dot(r, z)
            beta = rz_new / (rz_old + 1e-12)
            p = z + beta * p
            rz_old = rz_new
            num_iters = i + 1

        info = {
            "cg_residual": float(residual_norm.detach().cpu()),
            "cg_iters": int(num_iters),
            "cg_converged": bool(converged),
            "cg_initial_residual": float(b_norm.detach().cpu()),
        }

        if return_info:
            return x, info
        return x


class ShuffledRegression:
    """PyTorch variant of the shuffled regression helper class."""

    def __init__(
        self,
        x,
        y,
        a,
        b,
        epsilon: float,
        threshold: float,
        num_steps_sgd: int,
        sgd_learning_rate: float,
        n_s: int,
        num_steps_newton: int,
        improvement_abs_threshold: float,
        patience: int,
        newton_learning_rate: float,
        abs_threshold: float,
        gd_learning_rate: float,
        num_steps_gd: int,
        svd_thr: float,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.x = _to_tensor(x, device=self.device, dtype=self.dtype)
        self.y = _to_tensor(y, device=self.device, dtype=self.dtype)
        self.a = _to_tensor(a, device=self.device, dtype=self.dtype)
        self.b = _to_tensor(b, device=self.device, dtype=self.dtype)
        self.epsilon = float(epsilon)
        self.threshold = float(threshold)
        self.num_steps_sgd = int(num_steps_sgd)
        self.sgd_learning_rate = float(sgd_learning_rate)
        self.n = self.x.shape[0]
        self.n_s = int(n_s)
        self.num_steps_newton = int(num_steps_newton)
        self.improvement_abs_threshold = float(improvement_abs_threshold)
        self.patience = int(patience)
        self.newton_learning_rate = float(newton_learning_rate)
        self.abs_threshold = float(abs_threshold)
        self.gd_learning_rate = float(gd_learning_rate)
        self.num_steps_gd = int(num_steps_gd)
        self.svd_thr = float(svd_thr)
        self.final_newton_loss: Optional[float] = None
        self.sinkhorn = TorchSinkhornHessian(self.svd_thr, device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Basic building blocks

    def loss_value(self, params: torch.Tensor) -> torch.Tensor:
        y_pred = self.x @ params
        return self.sinkhorn.solve_ott_cost(
            y_pred,
            self.y,
            self.a,
            self.b,
            self.epsilon,
            self.threshold,
        )

    def value_and_grad(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, TorchOTResult]:
        params = params.requires_grad_(True)
        y_pred = self.x @ params
        ot = self.sinkhorn.solve_ott(y_pred, self.y, self.a, self.b, self.epsilon, self.threshold)
        (grads,) = torch.autograd.grad(ot.reg_ot_cost, params, retain_graph=True)
        params = params.detach()
        return ot.reg_ot_cost.detach(), grads.detach(), ot

    def value_and_grad_and_hess(self, params: torch.Tensor):
        value, grads, ot = self.value_and_grad(params)
        hess = self.sinkhorn.compute_hessian(ot)
        # Project Hessian onto parameter space using the chain rule.
        x_hess = torch.tensordot(self.x, hess, dims=([0], [0]))
        hess_w = torch.tensordot(x_hess, self.x, dims=([2], [0])).permute(0, 1, 3, 2)
        grads_matrix = grads.view(-1)
        hess_matrix = hess_w.reshape(grads_matrix.shape[0], grads_matrix.shape[0])
        dw = torch.linalg.solve(hess_matrix + self.svd_thr * torch.eye(hess_matrix.shape[0], device=self.device, dtype=self.dtype), grads_matrix)
        return value, grads, dw.view_as(grads)

    def hess_params(self, ot: TorchOTResult) -> torch.Tensor:
        hess = self.sinkhorn.compute_hessian(ot)
        x_hess = torch.tensordot(self.x, hess, dims=([0], [0]))
        return torch.tensordot(x_hess, self.x, dims=([2], [0])).permute(0, 1, 3, 2)

    def value_and_grad_sgd(self, params: torch.Tensor, indices: torch.Tensor):
        x_part = self.x[indices]
        a_part = self.a[indices]
        a_part = a_part / a_part.sum()
        params = params.requires_grad_(True)
        y_pred = x_part @ params
        ot = self.sinkhorn.solve_ott(y_pred, self.y, a_part, self.b, self.epsilon, self.threshold)
        (grads,) = torch.autograd.grad(ot.reg_ot_cost, params)
        return ot.reg_ot_cost.detach(), grads.detach()

    # ------------------------------------------------------------------
    # Training routines (kept intentionally simple)

    def fit(self, method: str, params_initial: torch.Tensor):
        params = params_initial.to(device=self.device, dtype=self.dtype)
        loss_list = []
        grads_list = []
        params_list = []

        if method not in {"SGD-Newton", "SGD-GD"}:
            raise ValueError(f"Unknown optimization method: {method}")

        value, grads, ot = self.value_and_grad(params)
        loss_list.append(value.detach().cpu().numpy())
        grads_list.append(grads.detach().cpu().numpy())
        params_list.append(params.detach().cpu().numpy())

        for _ in range(self.num_steps_sgd):
            params = params - self.sgd_learning_rate * grads
            indices = torch.randperm(self.n)[: self.n_s]
            value, grads = self.value_and_grad_sgd(params, indices)
            loss_list.append(value.detach().cpu().numpy())
            grads_list.append(grads.detach().cpu().numpy())
            params_list.append(params.detach().cpu().numpy())

        if method == "SGD-Newton":
            value, grads, ot = self.value_and_grad(params)
            self.final_newton_loss = float(value.detach().cpu())
            hess_params = self.hess_params(ot)
            hess_matrix = hess_params.reshape(grads.numel(), grads.numel())
            direction = torch.linalg.solve(
                hess_matrix + self.svd_thr * torch.eye(hess_matrix.shape[0], device=self.device, dtype=self.dtype),
                grads.view(-1),
            ).view_as(grads)
            params = params - self.newton_learning_rate * direction
            loss_list.append(value.detach().cpu().numpy())
            grads_list.append(grads.detach().cpu().numpy())
            params_list.append(params.detach().cpu().numpy())
        else:
            if self.final_newton_loss is None:
                raise ValueError("Run the 'SGD-Newton' stage before fine-tuning with gradient descent.")
            for _ in range(self.num_steps_gd):
                value, grads, _ = self.value_and_grad(params)
                params = params - self.gd_learning_rate * grads
                loss_list.append(value.detach().cpu().numpy())
                grads_list.append(grads.detach().cpu().numpy())
                params_list.append(params.detach().cpu().numpy())
                if abs(float(value.detach().cpu()) - self.final_newton_loss) < self.abs_threshold:
                    break

        return loss_list, grads_list, params_list

    def predict(self, params: torch.Tensor) -> torch.Tensor:
        return self.x @ params

    @staticmethod
    def parames_error(params_list, w):
        w_t = torch.as_tensor(w)
        return [torch.linalg.norm(torch.as_tensor(p) - w_t).item() for p in params_list]
