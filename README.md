# Project 2, TMA4180 Optimization 1

In this project we use BFGS with linesearch on a constrained optimization problem. 

Description of the files
main.py :
- Define constants and size of constraints.
- Choose classification type of A-matrix.
- Create z-list and find solution. Here you specifiy if you want to create a new random dataset, or use an existing one
  from z_list.npy. Each time you create a new dataset this will be saved into z_list.npy.
- Plotting

functions.py :
- Creating constraints and the gradient of the constraints
- Creating the data sets
- Computation of: f, P, grad(f), grad(P), lagrange

methods.py :
- backtrackingLinesearch() computes the step length alpha.
- primalBarrier() uses the BFGS method with step lengths form backtrackingLinesearch and stops when
  it is satisfied either by the KKT-conditions or small enough my.

plotting.py :
- evalute function value
- classify by ellipse/rectangle/with misclassification
- plot contour line
- plot z-points
