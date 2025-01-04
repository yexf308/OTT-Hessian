import jax
import numpy as np
import jax.numpy as jnp
from jax.example_libraries import optimizers as jax_opt
import optax

import time
import timeit

import ott
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.solvers.linear import implicit_differentiation as imp_diff


class SinkhornHessian:
    """Sinkhorn Hessian computation class with JAX."""
    def __init__(self, svd_thr):
        self.svd_thr = svd_thr 

    """ """
    @staticmethod
    def solve_ott( x, y, mu, nv, 𝜀, threshold):
        geom = pointcloud.PointCloud(x, y, epsilon=𝜀)
        prob = linear_problem.LinearProblem(geom, a=mu, b=nv)
        solver = sinkhorn.Sinkhorn(
            threshold=threshold, use_danskin=False, max_iterations=200000
            #solve_kwargs={
            #"implicit_diff": imp_diff.ImplicitDiff() if implicit else None}
        )
        out = solver(prob)
        return out
        #if compute_cost:
        #    return out.reg_ot_cost
        #else:
        #    return out

    @staticmethod
    def solve_ott_cost( x, y, mu, nv, 𝜀, threshold):
        geom = pointcloud.PointCloud(x, y, epsilon=𝜀)
        prob = linear_problem.LinearProblem(geom, a=mu, b=nv)
        solver = sinkhorn.Sinkhorn(
            threshold=threshold, use_danskin=False, max_iterations=200000, implicit_diff= None 
            #solve_kwargs={
            #"implicit_diff": imp_diff.ImplicitDiff() if implicit else None}
        )
        out = solver(prob)
        return out.reg_ot_cost
    
    @staticmethod   
    def solve_ott_implicit_cost( x, y, mu, nv, 𝜀, threshold):
        geom = pointcloud.PointCloud(x, y, epsilon=𝜀)
        prob = linear_problem.LinearProblem(geom, a=mu, b=nv)
        solver = sinkhorn.Sinkhorn(
            threshold=threshold, use_danskin=False, max_iterations=200000, 
            implicit_diff= imp_diff.ImplicitDiff() 
        )
        out = solver(prob)
        return out.reg_ot_cost
    
    
     
    
    @staticmethod
    def dOTdx(ot):
        x = ot.geom.x
        y = ot.geom.y
        dCk_dxk = 2 * (x[:, None, :] - y[None, :, :]) 
        P = ot.matrix
        grad = jnp.sum(dCk_dxk * P[:, :, None], axis=1)
        return grad
    
    @staticmethod
    def LHS_matrix(ot):
        a = ot.a
        b = ot.b
        P = ot.matrix
        a_P = jnp.sum(P, axis=1)
        b_P = jnp.sum(P, axis=0)

        a_diag = jnp.diag(a_P)
        b_diag = jnp.diag(b_P)
        PT = jnp.transpose(P)

        H1 = jnp.concatenate([a_diag, P], axis=1)
        H2 = jnp.concatenate([PT, b_diag], axis=1)
        H = jnp.concatenate([H1, H2], axis=0)

        return H
    

    
    @staticmethod
    def RHS(ot):
        x = ot.geom.x
        y = ot.geom.y
        dCk_dxk = 2 * (x[:, None, :] - y[None, :, :]) 
        P = ot.matrix
        b_g = jnp.transpose(dCk_dxk * P[:, :, None], [1, 0, 2])
        b_g_col = jnp.sum(b_g, axis=0)

        M, D = b_g_col.shape
        b_f = jnp.zeros((M, M, D))
        indices = (jnp.arange(M), jnp.arange(M))
        b_f = b_f.at[indices[0], indices[1], :].set(b_g_col)

        b = jnp.concatenate([b_f, b_g], axis=0)
        return b
    

    def compute_hessian_no_reg(self, ot): 
        epsilon = ot.geom.epsilon
        H   = self.LHS_matrix(ot)
        nm  = H.shape[0]
        R   = self.RHS(ot)
        m   = R.shape[1]
        dim = R.shape[2]

        R_reshape     = jnp.reshape(R, [nm, m*dim])
        HdagR_reshape = jnp.linalg.solve(H, R_reshape)
        HdagR         = jnp.reshape(HdagR_reshape, [nm,m,dim])
        Hessian_1     = jnp.einsum('skd, sjt->kdjt', R, HdagR)/epsilon

        x = ot.geom.x
        y = ot.geom.y
        P = ot.matrix
        dCk_dxk = 2 * (x[:, None, :] - y[None, :, :])
        d2Ck_dx2k = 2
        M, N, D = dCk_dxk.shape

        weighted_C = -dCk_dxk / epsilon * P[:, :, None]
        Hessian_2_part = jnp.einsum('kjs,kjt->kst', weighted_C, dCk_dxk)
        Hessian_3_diag = jnp.sum(d2Ck_dx2k * P, axis=1)

        identity_matrix = jnp.eye(D)
        expanded_Hessian_3_diag = Hessian_3_diag[:, None, None]
        G = Hessian_2_part + expanded_Hessian_3_diag * identity_matrix

        Hessian_2 = jnp.zeros((M, D, M, D))
        indices = jnp.arange(M)
        Hessian_2 = Hessian_2.at[indices, :, indices, :].set(G)

        Hessian = Hessian_1 + Hessian_2
        return Hessian


    def compute_hessian(self, ot):
        epsilon = ot.geom.epsilon
        H       = self.LHS_matrix(ot)
        R       = self.RHS(ot)

        # if svd_thr is larger than machine epsilon, then apply regularization automatically
    
        eigenvalues, eigenvectors = jnp.linalg.eigh(H)
        eigenvalues_sqrt_inv = jnp.where(eigenvalues > self.svd_thr, 1 / jnp.sqrt(eigenvalues), 0)
        Hsqrt = eigenvectors * eigenvalues_sqrt_inv[jnp.newaxis, :]
        bHsqrt = jnp.einsum('ikd, is ->ksd', R, Hsqrt)
        Hessian_1 = jnp.einsum('ksd, jst->kdjt', bHsqrt, bHsqrt) / epsilon

        x = ot.geom.x
        y = ot.geom.y
        P = ot.matrix
        dCk_dxk = 2 * (x[:, None, :] - y[None, :, :])
        d2Ck_dx2k = 2
        M, N, D = dCk_dxk.shape

        weighted_C = -dCk_dxk / epsilon * P[:, :, None]
        Hessian_2_part = jnp.einsum('kjs,kjt->kst', weighted_C, dCk_dxk)
        Hessian_3_diag = jnp.sum(d2Ck_dx2k * P, axis=1)

        identity_matrix = jnp.eye(D)
        expanded_Hessian_3_diag = Hessian_3_diag[:, None, None]
        G = Hessian_2_part + expanded_Hessian_3_diag * identity_matrix

        Hessian_2 = jnp.zeros((M, D, M, D))
        indices = jnp.arange(M)
        Hessian_2 = Hessian_2.at[indices, :, indices, :].set(G)

        Hessian = Hessian_1 + Hessian_2
        return Hessian
    

    def hess_loss_implicit(self, x,y,mu,nv, 𝜀, threshold):
        return jax.hessian(self.solve_ott_implicit_cost,  argnums=0)(x, y,mu,nv, 𝜀, threshold)
    

    def hess_loss_unroll(self, x, y,mu,nv, 𝜀, threshold):
        return jax.hessian(self.solve_ott_cost,  argnums=0)(x, y,mu,nv, 𝜀, threshold)
    

    def hess_loss_analytical(self, x,y,mu,nv, 𝜀, threshold):
        ot = self.solve_ott(x, y, mu, nv, 𝜀, threshold)
        return self.compute_hessian(ot)
    

    def hess_loss_analytical_no_reg(self, x,y,mu,nv, 𝜀, threshold):
        ot = self.solve_ott(x, y, mu, nv, 𝜀, threshold)
        return self.compute_hessian_no_reg(ot)
    
        

class ShuffledRegression:
  
    def __init__(self, x, y, a, b, 𝜀, threshold, num_steps_sgd, sgd_learning_rate, n_s, num_steps_newton, 
                 improvement_abs_threshold,Patience, newton_learning_rate, abs_threshold , gd_learning_rate, num_steps_gd, svd_thr):
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.𝜀  = 𝜀
        self.threshold = threshold
        self.num_steps_sgd = num_steps_sgd
        self.sgd_learning_rate = sgd_learning_rate
        self.n = x.shape[0]
        self.n_s = n_s
        self.num_steps_newton = num_steps_newton
        self.improvement_abs_threshold = improvement_abs_threshold
        self.Patience = Patience
        self.newton_learning_rate = newton_learning_rate
        self.abs_threshold = abs_threshold
        self.gd_learning_rate = gd_learning_rate
        self.num_steps_gd = num_steps_gd
        self.svd_thr = svd_thr
        self.final_newton_loss = None



    def loss_value(self, params):
        y_pred      = self.x @ params
        ot          = SinkhornHessian.solve_ott_cost(y_pred, self.y, self.a, self.b, self.𝜀, self.threshold)
        value       = ot.reg_ot_cost
        return value

    #@jax.jit
    def value_and_grad(self, params):
        y_pred      = self.x @ params
        ot          = SinkhornHessian.solve_ott(y_pred, self.y, self.a, self.b, self.𝜀, self.threshold)
        value       = ot.reg_ot_cost
        grads       = self.x.T@SinkhornHessian.dOTdx(ot)
        return value, grads, ot
    

    def value_and_grad_and_hess(self, params):
        value, grads, ot = self.value_and_grad(params)
        SH = SinkhornHessian(self.svd_thr)
        hess = SH.compute_hessian(ot)
        x_Hess = jnp.tensordot(self.x, hess, axes=((0),(0)))
        hess_w = jnp.tensordot(x_Hess, self.x,axes=((2),(0))).transpose([0,1,3,2])
        dw = jnp.linalg.tensorsolve(hess_w, grads)

        return value, grads, dw


    def hess_params(self, ot):
        SH = SinkhornHessian(self.svd_thr)
        hess = SH.compute_hessian(ot)
        x_Hess = jnp.tensordot(self.x, hess, axes=((0),(0)))
        hess_w = jnp.tensordot(x_Hess, self.x,axes=((2),(0))).transpose([0,1,3,2])
        return hess_w

    
 
    def value_and_grad_sgd(self, params, indices):
        
        x_part    = self.x[indices] 
        a_part    = self.a[indices]
        a_part    = a_part / jnp.sum(a_part)

        y_pred      = x_part @ params
        ot          = SinkhornHessian.solve_ott(y_pred, self.y, a_part, self.b, self.𝜀, self.threshold)
        value       = ot.reg_ot_cost
        grads       = x_part.T@SinkhornHessian.dOTdx(ot)
        return value, grads
    
    def opt_step(self, opt_state, optimizer, values, grads, params):
        updates, new_opt_state = optimizer.update(grads, opt_state)# , params,value=values, grad=grads, value_fn=self.loss_value)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

        
    
    def fit(self, method, params_intial):
        loss_list   = []
        grads_list  = []
        params_list = []
        
        """Initialize the optimizer"""
        optimizer_sgd = optax.chain(
                        optax.sgd(learning_rate=self.sgd_learning_rate)
                )
        
        """jit the functions"""
        value_and_grad_jit = jax.jit(self.value_and_grad)
        value_and_grad_and_hess_jit = jax.jit(self.value_and_grad_and_hess)
        hess_params_jit = jax.jit(self.hess_params)

        
        """Initialize the parameters"""
        params_size = jnp.size(params_intial)
        opt_state   = optimizer_sgd.init(params_intial)
        values, grads, ot = value_and_grad_jit(params_intial)
        params = params_intial
        print('Initial Cost:', values)

        loss_list.append(values)
        grads_list.append(grads)
        params_list.append(params)

        hess_w    = hess_params_jit(ot)
        eigv      = jnp.linalg.eigvalsh(hess_w.reshape(params_size,params_size))
        print('Initial Eigenvalues:', eigv)

        start_time                = time.time()
        stop_counter              = 0

        """SGD Method"""
        print('================================================================')
        print('Starting SGD method')
        for i in range( self.num_steps_sgd):
            """Update the parameters"""
            #params, opt_state = self.opt_step( opt_state, optimizer_sgd, grads, values, params)
            updates, opt_state = optimizer_sgd.update(grads, opt_state)# , params,value=values, grad=grads, value_fn=self.loss_value)
            params = optax.apply_updates(params, updates)
            np.random.seed(1)
            indices = np.random.choice (self.n, size=(self.n_s,), replace=False) 
            values, grads = self.value_and_grad_sgd(params, indices)

            loss_list.append(values)
            grads_list.append(grads)
            params_list.append(params)
            print('Step:', i, 'Cost:', values)

            """Check eigenvalues of Hessian"""
            if (i+1)% 5 ==0:
                values, grads, ot = value_and_grad_jit(params)
                loss_list[-1] = values
                hess_w    = hess_params_jit(ot)
                eigv      = jnp.linalg.eigvalsh(hess_w.reshape(params_size,params_size))
                print('Eigenvalues:', eigv)

                if jnp.all(eigv > 0):
                    print("All eigenvalues of Hessian for params are positive now")
                    break
            
        t1 = time.time()-start_time
        sgd_steps = i
        print('Time taken for SGD:', t1, 'Number of steps:', sgd_steps)

        if method == 'SGD-Newton':
            """Newton's method"""
            print('================================================================')
            print('Starting Newtons method')
            optimizer_newton = optax.chain(
                            optax.sgd(learning_rate=self.newton_learning_rate),
                    )
            params = params_list[-1]
            opt_state = optimizer_newton.init(params)

            start_time               = time.time()
            for j in range(self.num_steps_newton):
                values, grads, dw = value_and_grad_and_hess_jit(params)

                # # Compute the abs improvement
                if j > 0:  # Starting from the second iteration
                    previous_values = loss_list[-1]
                    abs_improvement = abs(previous_values - values)

                    # Check if improvement is less than the threshold
                    if abs_improvement < self.improvement_abs_threshold:
                        stop_counter += 1
                    else:
                        stop_counter = 0  # Reset counter if there is sufficient improvement

                    if stop_counter >= self.Patience:
                        print(f"Stopping early at iteration {j} due to lack of improvement.")
                        break

                # Update the parameters
                #params, opt_state = self.opt_step(opt_state, optimizer_newton, dw, values, params)
                updates, opt_state = optimizer_newton.update(dw, opt_state , params,value=values, grad=dw, value_fn=self.loss_value)
                params = optax.apply_updates(params, updates)


                loss_list.append(values)
                grads_list.append(grads)
                params_list.append(params)
                print('Step:', j, 'Cost:', values)
        
            t2           = time.time() - start_time
            newton_steps = j
            self.final_newton_loss = values
            print('Time taken for Newton:', t2, 'Number of steps:', newton_steps)
            
        elif method == 'SGD-GD':
            if self.final_newton_loss is None:
                raise ValueError('Run SGD-Newton first to get the final loss')
            else:
                """Gradient Descent"""
                print('================================================================')
                print('Starting Gradient Descent method')
                optimizer_gd = optax.chain(
                                optax.sgd(learning_rate=self.gd_learning_rate)
                        )
                params = params_list[-1]
                opt_state = optimizer_gd.init(params)

                start_time = time.time()
                for k in range(self.num_steps_gd):
                    values, grads, ot = value_and_grad_jit(params)

                    # Compute the abs improvement
                    if k > 0:
                        previous_values = loss_list[-1]
                        abs_improvement = abs(previous_values - values)
                    
                        # Check if improvement is less than the threshold
                        if abs_improvement < self.abs_threshold:
                            stop_counter += 1
                        else:
                            stop_counter = 0

                        if stop_counter >= self.Patience:
                            print(f"warm_start gd optimizer.")
                            stop_counter  = 0
                            opt_state              = optimizer_gd.init(params)

                    # Update the parameters
                    updates, opt_state = optimizer_gd.update(grads, opt_state)# , params,value=values, grad=grads, value_fn=self.loss_value)
                    params = optax.apply_updates(params, updates)

                    loss_list.append(values)
                    grads_list.append(grads)
                    params_list.append(params)
                    print('Step:', k, 'Cost:', values)

                    if abs(values - self.final_newton_loss) < self.abs_threshold:
                        print('Stopping early due to convergence')
                        break

        return loss_list, grads_list, params_list
    

    def predict(self, params):
        return self.x @ params
    
    @staticmethod
    def parames_error(params_list, w):
        error = []
        for params in params_list:
            error.append(jnp.linalg.norm(params-w))
        return error
    
    














        
    


    

    



        






    




   

