#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import functions as f

def backtrackingLinesearch(func, dfunc, z_list, n, p, x, my, lambda_low, lambda_high):
    alpha0 = 1
    rho = 0.5
    c1 = 0.4
    alpha = alpha0
    f0 = func(z_list, n, x, my, lambda_low, lambda_high)
    while True:
        print("Backtracking", alpha)
        c = f.c_function(x+alpha*p, lambda_low, lambda_high)
        if np.min(c) < 1e-10:        #Holder oss innenfor. Endra fra <= 0 til < TOL.
            alpha = rho * alpha
        elif func(z_list, n, x + alpha * p, my, lambda_low, lambda_high) <= f0 + c1 * alpha * np.dot(dfunc(z_list, n, x, my, lambda_low, lambda_high), p):
            return alpha
        else:
            alpha = rho * alpha

# Skifta navn fra BFGS til primalBarrier
def primalBarrier(func, dfunc, z_list, n, xk, lambda_low, lambda_high):
    my = 0.4
    tau = 1e-10    #grense for å stoppe BFGS
    I=np.identity(int(n * (n + 1) / 2) + n)
    dfk = dfunc(z_list, n, xk, my, lambda_low, lambda_high)
    while True:
        print("start ytterste while")
        Hk = I
        while np.linalg.norm(dfk, 2) > tau:
            print("innerste while")
            p = -Hk.dot(dfk) #Alltid descent for BFGS håper jeg. Tok vekk handling av nondescent direction.
            alpha = backtrackingLinesearch(func, dfunc, z_list, n, p, xk, my, lambda_low, lambda_high)
            if alpha < 1e-6:
                print("alpha", alpha)
                break
            xk_prev = xk
            xk = xk + alpha*p
            sk = xk - xk_prev
            dfk_prev = dfk
            dfk = dfunc(z_list, n, xk, my, lambda_low, lambda_high)
            yk = dfk - dfk_prev
            Hk_prev = Hk
            if np.dot(yk, sk) > 0: #Update the Hessian if ok
                rho = 1 / np.dot(yk, sk)
                Hk = np.matmul(I-rho*np.outer(sk,yk),np.matmul(Hk_prev,I-rho*np.outer(yk,sk))) + rho*np.outer(sk,sk)
#        return xk

        zk = my/f.c_function(xk, lambda_low, lambda_high)
        columnvector = np.array([zk[0]-zk[1]+zk[4]*xk[2], -2*zk[4]*xk[1], zk[2]-zk[3]+zk[4]*xk[0], 0, 0])
        if np.max((dfk - columnvector)) < 1e-10 and my < 1e-10: #KKT conditions
            print("KKT", dfk - columnvector)
            return xk
        my = 0.5*my
        # Fix initializing of all stuff
