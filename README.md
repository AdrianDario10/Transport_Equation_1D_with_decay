# Transport Equation with decay 1D
Physics informed neural network (PINN) for the 1D Transport equation with decay

This module implements the Physics Informed Neural Network (PINN) model for the 1D Transport equation with decay. The Transport equation with decay is given by (d/dt - c d/dx -1/2)u = 0, where c is 1. This initial value problem has an initial condition u(t=0, x) =  1/(1+x**2). The PINN model predicts u(t, x) for the input (t, x).

The effectiveness of PINNs is validated in the following works.

+  M. Raissi, et al., Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations, arXiv: 1711.10561 (2017). (https://arxiv.org/abs/1711.10561)

+  M. Raissi, et al., Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations, arXiv: 1711.10566 (2017). (https://arxiv.org/abs/1711.10566)

It is based on hysics informed neural network (PINN) for the 1D Wave equation on https://github.com/okada39/pinn_wave
