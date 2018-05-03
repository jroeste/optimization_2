#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import methods as meth
import functions as f
import plotting as plot
import sys



'''Constants:'''
m = 50 # number of points
n = 2 # dimensions
area = 2.0
x_length = int(n*(n+1)/2)+n
prob = 0.05
min_rec, max_rec = 1, 4


lambda_low = 0.00001
lambda_high = 10

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
        A=[[1, 0.5], [0.5, 1]]    #Makes a classifying ellipse. [[1, 0.5], [0.5, 0.5]] er eksempel på
                                    # testproblem der løsningen er grei, f eks [[0.001, 3], [3, 0.005]] (og lignende)
                                    #er målet å få til å funke
        b=[0,0]
        z_list = f.construct_z_elliptic(n, m, A, b, area)
        solution = meth.primalBarrier(f.P, f.dP, z_list, n, x_initial, lambda_low, lambda_high)
        #print(solution)

        # The rest here is for plotting
        A,b=f.construct_A_and_b(n, solution)
        X, Y, Z = plot.make_ellipse(A, b, area, plot.eval_func_model_2D)
        plot.plot_dataset_2d(X, Y, Z)
        plot.plot_z_points(z_list, m)
        plt.show()


    elif Master_Flag=='Plot':
        print("hi there")

    elif Master_Flag=='Testing functions':
        x=np.linspace(-5,5,m)

