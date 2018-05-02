#!/usr/bin/env python
# -*- coding: utf-8 -*-

#denne funksjonen er som før
def note3algoritme(func, dfunc, z_list, n, p, x, my, lambda_low, lambda_high):
    c1 = 0.5
    c2 = 0.6
    alpha = 1
    alpha_min = 0
    alpha_max = np.inf
    dfk = dfunc(z_list, n, x, my, lambda_low, lambda_high)
    fk=func(z_list, n, x, my, lambda_low, lambda_high)
    while True:
        c=f.c_function(x+alpha*p, lambda_low, lambda_high)
        if np.min(c)<=0:        #Her er det lagt til for ikkje å hoppe utenfor "lovlig" område
            alpha_max = alpha
            alpha = (alpha_max + alpha_min) / 2

        elif func(z_list, n, x + alpha*p, my, lambda_low, lambda_high) > fk + c1*alpha*np.dot(dfk, p):  #No suff decr
            alpha_max = alpha
            alpha = (alpha_max + alpha_min)/2
        elif np.dot(dfunc(z_list, n, x + alpha*p, my, lambda_low, lambda_high), p) < c2*np.dot(dfk, p): # No curv con
            alpha_min = alpha
            if np.isinf(alpha_max):
                alpha = 2*alpha
            else:
                alpha = (alpha_max + alpha_min) / 2
        else:
            return alpha

#Denne er som før
def steepestDescent(f, df, z_list, n, xk, my, lambda_low, lambda_high):
    fk = f(z_list, n, xk)
    dfk = df(z_list, n, xk)
    residuals = []
    residuals.append(fk)
    while fk > 10e-4 and np.linalg.norm(dfk, 2) > 10e-6:
        p = - dfk
        print("Er i steepest descent og bruker backtracking")
        alpha = note3algoritme(f, df, z_list, n, p, xk, my, lambda_low, lambda_high)
        xk = xk + alpha * p
        fk=f(z_list, n, xk)
        dfk=df(z_list, n, xk)
        residuals.append(fk)
    return xk, residuals

#Tatt fra BFGS:
if np.dot(p, dfk) > 0:  # Her la vi til sjekk av retning som descent-retning
    print("Resetting to steepest descent")  # Dette var vel strengt tatt kun nødvendig for Fletcher Reeves
    p = -dfk
