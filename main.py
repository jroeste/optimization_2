#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import methods as meth
import functions as f
import plotting as plot
import sys



'''Constants:'''
m = 2000 # number of points
n = 2 # dimensions
area = 2.0
x_length = int(n*(n+1)/2)+n
prob = 0.05
min_rec, max_rec = 1, 4


lambda_low = 1
lambda_high = 10

# Create a feasible starting point
x_initial = np.zeros(x_length)
x_initial[0], x_initial[2] = (lambda_low+lambda_high)/2, (lambda_low+lambda_high)/2
x_initial[1] = np.sqrt((x_initial[0]*x_initial[2]-lambda_low**2)/2)
print("x_initial", x_initial)

if np.amin(f.c_function(x_initial, lambda_low, lambda_high)) < 1e-20:
    sys.exit("Illegal starting point")

if __name__ == "__main__":

    Master_Flag = {
                    0: 'Create dataset',
                    1: 'Plot',
                    2: 'Testing functions',




            }[0]
    if Master_Flag =='Create dataset':
        # Define the classifying ellipse.

        A_classification = [[1, 0], [0, 1]]  # Circle
        A_classification = [[1, 0.3], [0.3, 0.4]]  # Ellipse
        #A_classification = [[0.0001, 10], [10, 0.0001]]  # Hyperbola
        #A_classification = [[0.00001, 50], [50, 0.00001]]  # Hyperbola closer to asymptotes
        A_classification = [[0.1, 10], [10, 0.1]]  # A third hyperbola where KKT is not fulfilled


        b_classification=[0, 0]
        z_list = f.construct_z_elliptic(n, m, A_classification, b_classification, area)
        solution = meth.primalBarrier(f.P, f.dP, z_list, n, x_initial, lambda_low, lambda_high)

        # The rest here is for plotting
        A_solution,b_solution=f.construct_A_and_b(n, solution)
        plot.plot_z_points(z_list, m)
        X, Y, Z = plot.make_ellipse(A_solution, b_solution, area, plot.eval_func_model_2D)
        plot.plot_dataset_2d(X, Y, Z)
        print("\neigenvalues of classification matrix\n", np.linalg.eigvals(A_classification))
        plt.show()


    elif Master_Flag=='Plot':
        print("hi there")

    elif Master_Flag=='Testing functions':
        x=np.linspace(-5,5,m)

