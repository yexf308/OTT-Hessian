This is demo code to generate the results in "Robust First and Second-Order Differentiation for Regularized
Optimal Transport" (https://arxiv.org/pdf/2407.02015)

*util.py*: utility functions. 

*SinkhornHessian.py*: Main function classes -- SinkhornHessian and ShuffledRegression.
*torch_sinkhorn_hessian.py*: PyTorch implementation of the Sinkhorn Hessian utilities mirroring the JAX API.

*CondH.py*: It plots the delay of the smallest positive eigenvalue $\lambda_{2N-1}$ in $N$ and $\epsilon$ in two cases: (1) Equally spaced points on the unique circle and (2)  Uniformly distributed point cloud in unit square $[0,1]^2$.  (Figure 1)

*Hessian_speed_uniform.py*: It plots the comparison of runtime (in seconds) in Hessian computing among three approaches: unroll, implicit differentiation and analytic expression with regularization (ours). (Figure 2 Top)

*Hessian_accuracy_uniform.py*: It plots the comparison of marginal error for Hessian computing among three approaches: unroll, implicit differentiation and analytic expression with regularization (ours). (Figure 2 Bottom)

*shuffled_regression.py*: It plots the results in shuffled regression with Gaussian mixtures. (Figure 3)
  Use the ``--backend`` flag to switch between the JAX (default) and PyTorch
  implementations.

*3D_pointcloud_regist.py*: It plots the results in 3D Point Cloud Registration. (Figure 4) The ModelNet10 dataset can be downloaded in https://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip. 


