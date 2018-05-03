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
x_initial=np.zeros(x_length)
x_initial[0], x_initial[2] = (lambda_low+lambda_high)/2, (lambda_low+lambda_high)/2
x_initial[1] = np.sqrt((x_initial[0]*x_initial[2]-lambda_low**2)/2)
print "x_initial", x_initial

if np.amin(f.c_function(x_initial, lambda_low, lambda_high)) < 1e-20:
    sys.exit("Illegal starting point")

if __name__ == "__main__":

    Master_Flag = {
                    0: 'Create dataset',
                    1: 'Plot',
                    2: 'Testing functions',




            }[0]
    if Master_Flag =='Create dataset':
        A_classification = [[0.0001, 30], [30, 0.0001]]
        #A_classification = [[1, 0], [0, 1]] #Makes a classifying ellipse.
        print "eigenvalues of classification matrix\n", np.linalg.eigvals(A_classification)


        b_classification=[0, 0]
        z_list = f.construct_z_elliptic(n, m, A_classification, b_classification, area)
        solution = meth.primalBarrier(f.P, f.dP, z_list, n, x_initial, lambda_low, lambda_high)
        #print(solution)

        # The rest here is for plotting
        A_solution,b_solution=f.construct_A_and_b(n, solution)
        X, Y, Z = plot.make_ellipse(A_solution, b_solution, area, plot.eval_func_model_2D)
        plot.plot_dataset_2d(X, Y, Z)
        plot.plot_z_points(z_list, m)
        print "\neigenvalues of classification matrix\n", np.linalg.eigvals(A_classification)
        plt.show()


    elif Master_Flag=='Plot':
        print("hi there")

    elif Master_Flag=='Testing functions':
        x=np.linspace(-5,5,m)

