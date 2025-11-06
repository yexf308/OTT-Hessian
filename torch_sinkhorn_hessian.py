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

import torch


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
    """Squared Euclidean ground cost used by the Sinkhorn solver."""

    diff = x[:, None, :] - y[None, :, :]
    return 0.5 * torch.sum(diff * diff, dim=-1)


def _sinkhorn(
    x: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    epsilon: float,
    threshold: float,
    max_iterations: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Balanced Sinkhorn iterations returning cost and transport plan.

    The routine operates entirely in PyTorch so that gradients with respect to
    the input point clouds can flow through the solver. The multiplicative
    updates follow the classical Sinkhorn-Knopp algorithm without log-space
    stabilization which is sufficient for the small problems used in the
    regression tests.
    """

    cost_matrix = _compute_cost_matrix(x, y)
    kernel = torch.exp(-cost_matrix / epsilon)

    u = torch.ones_like(mu)
    v = torch.ones_like(nu)

    iterations = 0
    for iterations in range(1, max_iterations + 1):
        u_prev = u
        Kv = torch.matmul(kernel, v)
        u = mu / (Kv + 1e-12)
        KT_u = torch.matmul(kernel.t(), u)
        v = nu / (KT_u + 1e-12)
        if torch.max(torch.abs(u - u_prev)).item() <= threshold:
            break

    transport_plan = u[:, None] * kernel * v[None, :]
    transport_cost = torch.sum(transport_plan * cost_matrix)
    entropy_term = torch.sum(transport_plan * (torch.log(transport_plan + 1e-12) - 1.0))
    reg_cost = transport_cost + epsilon * entropy_term

    return reg_cost, transport_plan, cost_matrix, iterations


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
    """

    cost_matrix = _compute_cost_matrix(x, y)
    kernel = torch.exp(-cost_matrix / epsilon)

    u = torch.ones_like(mu)
    v = torch.ones_like(nu)

    threshold_tensor = torch.tensor(threshold, dtype=x.dtype, device=x.device)
    iteration_ids = torch.arange(
        1, max_iterations + 1, dtype=torch.int64, device=x.device
    )
    recorded_iteration = torch.full((), max_iterations, dtype=torch.int64, device=x.device)
    has_converged = torch.zeros((), dtype=torch.bool, device=x.device)
    final_residual = torch.full((), float("inf"), dtype=x.dtype, device=x.device)

    for idx in range(max_iterations):
        u_prev = u
        Kv = torch.matmul(kernel, v)
        u = mu / (Kv + 1e-12)
        KT_u = torch.matmul(kernel.t(), u)
        v = nu / (KT_u + 1e-12)

        diff = torch.max(torch.abs(u - u_prev))
        converged_now = diff <= threshold_tensor
        update_mask = torch.logical_and(~has_converged, converged_now)
        recorded_iteration = torch.where(update_mask, iteration_ids[idx], recorded_iteration)
        has_converged = torch.logical_or(has_converged, converged_now)
        final_residual = diff

    transport_plan = u[:, None] * kernel * v[None, :]
    transport_cost = torch.sum(transport_plan * cost_matrix)
    entropy_term = torch.sum(transport_plan * (torch.log(transport_plan + 1e-12) - 1.0))
    reg_cost = transport_cost + epsilon * entropy_term

    return reg_cost, transport_plan, cost_matrix, recorded_iteration, final_residual


_COMPILED_SINKHORN = None
_COMPILED_MAX_ITERS = 64


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
    ) -> None:
        self.svd_thr = svd_thr
        self.max_iterations = max_iterations
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        if use_compile is None:
            use_compile = getattr(torch, "compile", None) is not None
        if use_compile:
            sinkhorn_impl = _get_compiled_sinkhorn()
        else:
            sinkhorn_impl = _sinkhorn
        self.use_compile = sinkhorn_impl is not _sinkhorn
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        result = self._sinkhorn(
            x,
            y,
            mu,
            nu,
            epsilon,
            threshold,
            max_iterations,
        )
        if len(result) == 5:
            cost, plan, cost_matrix, iterations, residual = result
        else:
            cost, plan, cost_matrix, iterations = result
            residual = None
        if isinstance(iterations, torch.Tensor):
            iterations_value = int(iterations.detach().cpu().item())
        else:
            iterations_value = int(iterations)
        return cost, plan, cost_matrix, iterations_value, residual

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
        cost, plan, _, iterations_value, residual = self._run_sinkhorn(
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
            cost, plan, _, iterations_value, _ = self._run_sinkhorn(
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
    # Differential quantities

    def dOTdx(self, ot: TorchOTResult) -> torch.Tensor:
        x = ot.geom.x.clone().detach().requires_grad_(True)
        cost, _, _, _, _ = self._run_sinkhorn(
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
        return self.compute_hessian(ot)

    def compute_hessian(self, ot: TorchOTResult) -> torch.Tensor:
        def loss_fn(x_tensor: torch.Tensor) -> torch.Tensor:
            cost, _, _, _, _ = self._run_sinkhorn(
                x_tensor,
                ot.geom.y,
                ot.a,
                ot.b,
                ot.geom.epsilon,
                ot.threshold,
                self.max_iterations,
            )
            return cost

        return torch.autograd.functional.hessian(loss_fn, ot.geom.x)

    def hess_loss_implicit(self, x, y, mu, nv, epsilon: float, threshold: float) -> torch.Tensor:
        y_t = _to_tensor(y, device=self.device, dtype=self.dtype)
        mu_t = _to_tensor(mu, device=self.device, dtype=self.dtype)
        nv_t = _to_tensor(nv, device=self.device, dtype=self.dtype)
        eps = float(epsilon)

        def loss_fn(x_tensor: torch.Tensor) -> torch.Tensor:
            cost, _, _, _, _ = self._run_sinkhorn(
                x_tensor,
                y_t,
                mu_t,
                nv_t,
                eps,
                threshold,
                self.max_iterations,
            )
            return cost

        x_tensor = _to_tensor(x, device=self.device, dtype=self.dtype)
        return torch.autograd.functional.hessian(loss_fn, x_tensor)

    def hess_loss_unroll(self, x, y, mu, nv, epsilon: float, threshold: float) -> torch.Tensor:
        return self.hess_loss_implicit(x, y, mu, nv, epsilon, threshold)

    def hess_loss_analytical(self, x, y, mu, nv, epsilon: float, threshold: float) -> torch.Tensor:
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
        ot = self.solve_ott(x, y, mu, nv, epsilon, threshold)
        return self.compute_hessian_no_reg(ot)


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
        loss_list.append(value.cpu().numpy())
        grads_list.append(grads.cpu().numpy())
        params_list.append(params.cpu().numpy())

        for _ in range(self.num_steps_sgd):
            params = params - self.sgd_learning_rate * grads
            indices = torch.randperm(self.n)[: self.n_s]
            value, grads = self.value_and_grad_sgd(params, indices)
            loss_list.append(value.cpu().numpy())
            grads_list.append(grads.cpu().numpy())
            params_list.append(params.cpu().numpy())

        if method == "SGD-Newton":
            value, grads, ot = self.value_and_grad(params)
            self.final_newton_loss = float(value.cpu())
            hess_params = self.hess_params(ot)
            hess_matrix = hess_params.reshape(grads.numel(), grads.numel())
            direction = torch.linalg.solve(
                hess_matrix + self.svd_thr * torch.eye(hess_matrix.shape[0], device=self.device, dtype=self.dtype),
                grads.view(-1),
            ).view_as(grads)
            params = params - self.newton_learning_rate * direction
            loss_list.append(value.cpu().numpy())
            grads_list.append(grads.cpu().numpy())
            params_list.append(params.cpu().numpy())
        else:
            if self.final_newton_loss is None:
                raise ValueError("Run the 'SGD-Newton' stage before fine-tuning with gradient descent.")
            for _ in range(self.num_steps_gd):
                value, grads, _ = self.value_and_grad(params)
                params = params - self.gd_learning_rate * grads
                loss_list.append(value.cpu().numpy())
                grads_list.append(grads.cpu().numpy())
                params_list.append(params.cpu().numpy())
                if abs(float(value.cpu()) - self.final_newton_loss) < self.abs_threshold:
                    break

        return loss_list, grads_list, params_list

    def predict(self, params: torch.Tensor) -> torch.Tensor:
        return self.x @ params

    @staticmethod
    def parames_error(params_list, w):
        w_t = torch.as_tensor(w)
        return [torch.linalg.norm(torch.as_tensor(p) - w_t).item() for p in params_list]

