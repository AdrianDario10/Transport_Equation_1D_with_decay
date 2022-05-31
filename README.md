# transport_equation_1D
Physics informed neural network (PINN) for the 1D Transport equation
This module implements the Physics Informed Neural Network (PINN) model for the 1D Transport equation. The Transport equation is given by (d/dt - c d/dx)u = 0, where c is 1. This initi value problem has an initial condition u(t=0, x) =  1/(1+x**2). The PINN model predicts u(t, x) for the input (t, x).
