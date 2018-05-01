import numpy as np
import matplotlib.pyplot as plt
import functions as f
import time
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

#Denne er for det meste som før
def BFGS(func, dfunc, z_list, n, xk, tau, my, lambda_low, lambda_high):
    residuals = []
    I=np.identity(int(n * (n + 1) / 2) + n)
    Hk = I
    fk = func(z_list, n, xk, my, lambda_low, lambda_high)
    residuals.append(fk)
    dfk = dfunc(z_list, n, xk, my, lambda_low, lambda_high)
    counter=0
    while fk > 10e-6 and np.linalg.norm(dfk, 2) > tau:
        #print(xk)
        p = -Hk.dot(dfk)
        if np.dot(p,dfk) >0:        #Her la vi til sjekk av retning som descent-retning
            print("Resetting to steepest descent")
            p = -dfk
        alpha = note3algoritme(func, dfunc, z_list, n, p, xk, my, lambda_low, lambda_high) #Skal ha backtracking?
        xk_prev=xk
        xk = xk + alpha*p
        sk = xk - xk_prev
        fk=func(z_list, n, xk, my, lambda_low, lambda_high)
        dfk_prev=dfk
        dfk = dfunc(z_list, n, xk, my, lambda_low, lambda_high)
        yk = dfk - dfk_prev
        rho = 1 / np.dot(yk, sk)
        Hk_prev=Hk
        Hk=np.matmul(I-rho*np.outer(sk,yk),np.matmul(Hk_prev,I-rho*np.outer(yk,sk))) + rho*np.outer(sk,sk)
        residuals.append(fk)
    return xk, residuals

#Denne er laga som algoritmen i boka
def log_barrier(z_list, n, xk ,lambda_low, lambda_high):
    my=0        #Setter my_0
    tau=1e-5    #grense for å stoppe BFGS
    while True:
        #print("c:", f.c_function(xk, lambda_low, lambda_high))
        xk_plus_one, res=BFGS(f.P, f.dP, z_list, n, xk, my, tau, lambda_low, lambda_high) #Beste løsning av f-log(div) for gitt my. Her kan P byttes ut med F-model for å få gammel kode.
        #print("test")
        if my<0.01:     #naivt stoppested
            return xk_plus_one
        my*=0.5 #reduser my før det heile gjentas
        xk=xk_plus_one      #oppdaterer start-x
