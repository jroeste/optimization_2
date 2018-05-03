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
        c = f.c_function(x+alpha*p, lambda_low, lambda_high)
        if np.amin(c) < 1e-10:        #keeps us inside the constraints
            alpha = rho * alpha
        elif func(z_list, n, x + alpha * p, my, lambda_low, lambda_high) <= f0 + c1 * alpha * np.dot(dfunc(z_list, n, x, my, lambda_low, lambda_high), p):
            return alpha
        else:
            alpha = rho * alpha

def primalBarrier(func, dfunc, z_list, n, xk, lambda_low, lambda_high):
    my = 1
    tau = 1e-5    #Stopping criteria for BFGS
    I=np.identity(int(n * (n + 1) / 2) + n)
    while True:
        #begin BFGS
        Hk = I
        dfk = dfunc(z_list, n, xk, my, lambda_low, lambda_high)
        while np.linalg.norm(dfk, 2) > tau:
            print("innerste while")
            p = -Hk.dot(dfk)/np.linalg.norm(Hk.dot(dfk),2)
            alpha = backtrackingLinesearch(func, dfunc, z_list, n, p, xk, my, lambda_low, lambda_high)
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
            if alpha < 1e-7:
                print("Breaking on too small alpha", alpha)
                break
        # end BFGS

        zk = my/f.c_function(xk, lambda_low, lambda_high)
        columnvector = np.array([zk[0]-zk[1]+zk[4]*xk[2], -2*zk[4]*xk[1], zk[2]-zk[3]+zk[4]*xk[0], 0, 0])
        print "grad p:", dfk
        print "grad f:", f.df_model(z_list, n, xk, my, lambda_low, lambda_high)
        print "columnvector", columnvector
        print "c(x)", f.c_function(xk, lambda_low, lambda_high)
        print "grad c(x)", f.dc_function(xk)
        if np.linalg.norm((f.df_model(z_list, n, xk, my, lambda_low, lambda_high) - columnvector), 2) < 1e-3 and my < 1e-3:
            print "\nKKT fulfilled"
            print "f value", f.f_model(z_list, n, xk, my, lambda_low, lambda_high)
            print "terminal x", xk
            A_final = f.construct_A_and_b(n, xk)[0]
            print "terminal matrix A\n", A_final
            print "with eigenvalues\n", np.linalg.eigvals(A_final)
            return xk
        # elif np.linalg.norm(dfk, 2) < 1e-6 and my < 1e-6:
        #     print "\nAlternative stopping criteria"
        #     print "f value", f.f_model(z_list, n, xk, my, lambda_low, lambda_high)
        #     print "terminal x", xk
        #     A_final = f.construct_A_and_b(n, xk)[0]
        #     print "terminal matrix A\n", A_final
        #     print "with eigenvalues\n", np.linalg.eigvals(A_final)
        #     return xk
        elif my < 1e-10:
            print "\nmy < 1e-10"
            print "f value", f.f_model(z_list, n, xk, my, lambda_low, lambda_high)
            print "terminal x", xk
            A_final = f.construct_A_and_b(n, xk)[0]
            print "terminal matrix A\n", A_final
            print "with eigenvalues\n", np.linalg.eigvals(A_final)
            return xk
        my = 0.5*my
        print "downscaling my to", my
