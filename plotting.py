#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

'''evaluate function value'''
def eval_func_model_2D(X,Y,A,b):
    return A[0][0]*X**2+2*A[0][1]*X*Y+A[1][1]*Y**2+b[0]*X+b[1]*Y

def classify_by_ellipse(m,n,area):
    a00=np.random.uniform(0.5,2)
    a11=np.random.uniform(0.5,2)
    minval = min(a00,a11)
    a01=np.random.uniform(0, minval)
    A = [[a00,a01], [a01, a11]]  #symmetric, positive definite A
    c = np.random.uniform(-1, 1, n) #random vector
    z = np.zeros((m,n+1))
    for i in range(m):
        z[i]=np.random.uniform(-area,area,n+1)

    '''Perform classification:'''
    for i in range(m):
        f_value=eval_func_model_2D(z[i][1],z[i][2],A,c)
        if f_value>=1:  #if outside the ellipse, the weight should be -1
            z[i][0]=-1
        else:
            z[i][0]=1   #if inside the ellipse, the weight should be +1
    return z

def classify_by_rectangle(m,n,area,min,max):
    rec = [-area / np.random.uniform(min, max)
        , area / np.random.uniform(min, max)
        , -area / np.random.uniform(min, max)
        , area / np.random.uniform(min, max)]
    z = np.zeros((m, n + 1))
    '''Perform classification:'''
    for i in range(m):
        z[i] = np.random.uniform(-area, area, n + 1)
        x = z[i][1]
        y = z[i][2]
        if rec[0] < x < rec[1] and rec[2]< y < rec[3]:
            z[i][0] = 1
        else:
            z[i][0] = -1
    return z

def classify_misclassification(m,n,area,prob):
    z_list=classify_by_ellipse(m,n,area)
    for i in range(m):
        a=np.random.uniform()
        if a<prob:
            z_list[i][0]*=-1
    return z_list


def plot_dataset_2d(X,Y,Z):
    CS = plt.contour(X, Y, Z, [1], linewidths=4, zorder=10)
    plt.clabel(CS, inline=1, fontsize=10)

def plot_z_points(z,m):
    for i in range(m):
        if z[i][0]<0:
            col='green'
        else:
            col='red'
        plt.plot(z[i][1], z[i][2], 'o', color=col, zorder=0)
    #plt.title(title)

def make_ellipse(A,b,area,func):
    delta = 0.01
    x = np.arange(-area*1.2, 1.2*area+delta, delta)
    y = np.arange(-area*1.2, 1.2*area+delta, delta)
    X, Y = np.meshgrid(x, y)
    Z=func(X, Y, A, b)
    return X,Y,Z