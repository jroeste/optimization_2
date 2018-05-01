import numpy as np
import matplotlib.pyplot as plt
import functions as f
import time

def note3algoritme(f, df, z_list, n, p, x):
    c1 = 0.5
    c2 = 0.6
    alpha = 1
    alpha_min = 0
    alpha_max = np.inf
    dfk = df(z_list, n, x)
    fk=f(z_list, n, x)
    while True:
        if f(z_list, n, x + alpha*p) > fk + c1*alpha*np.dot(dfk, p):  #No suff decr
            alpha_max = alpha
            alpha = (alpha_max + alpha_min)/2
        elif np.dot(df(z_list, n, x + alpha*p), p) < c2*np.dot(dfk, p): # No curv con
            alpha_min = alpha
            if np.isinf(alpha_max):
                alpha = 2*alpha
            else:
                alpha = (alpha_max + alpha_min) / 2
        else:
            return alpha


def steepestDescent(f, df, z_list, n, xk):
    fk = f(z_list, n, xk)
    dfk = df(z_list, n, xk)
    residuals = []
    residuals.append(fk)
    while fk > 10e-4 and np.linalg.norm(dfk, 2) > 10e-6:
        p = - dfk
        alpha = note3algoritme(f, df, z_list, n, p, xk)
        xk = xk + alpha * p
        fk=f(z_list, n, xk)
        dfk=df(z_list, n, xk)
        residuals.append(fk)
    return xk, residuals


def BFGS(f, df, z_list, n, xk, tau, my):
    residuals = []
    residuals.append(f(z_list, n, xk))
    I=np.identity(int(n * (n + 1) / 2) + n)
    Hk = I
    fk = f(z_list, n, xk, my)
    dfk = df(z_list, n, xk, my)
    counter=0
    while fk > 10e-2 and np.linalg.norm(dfk, 2) > tau:
        p = -Hk.dot(dfk)
        alpha = note3algoritme(f, df, z_list, n, p, xk)
        xk_prev=xk
        xk = xk + alpha*p
        sk = xk - xk_prev
        fk=f(z_list, n, xk, my)
        dfk_prev=dfk
        dfk = df(z_list, n, xk, my)
        yk = dfk - dfk_prev
        rho = 1 / np.dot(yk, sk)
        Hk_prev=Hk
        Hk=np.matmul(I-rho*np.outer(sk,yk),np.matmul(Hk_prev,I-rho*np.outer(yk,sk))) + rho*np.outer(sk,sk)
        residuals.append(fk)
    return xk, residuals


def log_barrier(z_list, n, xk):
    my=1
    tau=1e-2
    while True:
        xk_plus_one, res=BFGS(f.f_model, f.df_model, z_list, n, xk, my, tau)
        if my<0.001:
            return xk_plus_one
        my*=0.5
        xk=xk_plus_one