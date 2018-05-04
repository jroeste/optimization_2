import numpy as np
import matplotlib.pyplot as plt
import time
#Denne er ny, rekner ut c (constrains) som vektor
def c_function(x, lambda_low, lambda_high):
    c=np.zeros(5)
    c[0]=x[0] - lambda_low
    c[1]=-x[0] + lambda_high
    c[2]=x[2] - lambda_low
    c[3]=-x[2] + lambda_high
    c[4]=x[0]*x[2] - lambda_low**2-x[1]**2
    return c
#Denne er ny, rekner ut dc som 2d-vektor [dc/dx1, dc/dx2..]
def dc_function(x):
    grad_c=np.zeros((5,5))
    grad_c[0]=[1,0,0,0,0]
    grad_c[1]=[-1,0,0,0,0]
    grad_c[2]=[0,0,1,0,0]
    grad_c[3]=[0,0,-1,0,0]
    grad_c[4]=[x[2],-2*x[1],x[0],0,0]
    return grad_c

#Denne er som før
def f_model(z_list,n,x, my, lambda_low, lambda_high):
    A,b=construct_A_and_b(n,x)  #endre på construct??
    functionsum=0
    for i in range(len(z_list)):    #length m
        functionsum+=compute_r_i(z_list[i],A,b)**2
    return functionsum

#Denne finner f fra gammel funksjon, og trekker fra logleddet etterpå
def P(z_list, n, x, my, lambda_low, lambda_high):
    functionsum=f_model(z_list, n, x, my, lambda_low, lambda_high)
    c=c_function(x, lambda_low, lambda_high)
    for i in range(len(c)):
        functionsum-=my*np.log(c[i])
    return functionsum





def lagrange_z(my,x,lambda_low,lambda_high):
    return my/c_function(x,lambda_low,lambda_high)

#Denne er som før                   return A[0][0]*X**2+2*A[0][1]*X*Y+A[1][1]*Y**2+b[0]*X+b[1]*Y
def compute_r_i(z_list_i,A,b):
    if z_list_i[0]>0:
        return max([np.dot(z_list_i[1:],np.matmul(A,z_list_i[1:]))+np.dot(b,z_list_i[1:])-1,0])
    else:
        return max([1-np.dot(z_list_i[1:],np.matmul(A,z_list_i[1:]))-np.dot(b,z_list_i[1:]),0])

#Denne er som før
def construct_A_and_b(n,x):
    C=x[int(n*(n+1)/2):]
    A=np.zeros((n,n))
    index=0
    for h in range(n):
        for j in range(h,n):
            A[h][j] = x[index]
            A[j][h] = x[index]
            index+=1
    return A, C

#Denne er som før
def df_model(z_list,n,x, my, lambda_low, lambda_high):
    A,b = construct_A_and_b(n,x)
    dfx=np.zeros(int(n*(n+1)/2)+n)
    for i in range(len(z_list)):     #length m
        index = 0
        ri=compute_r_i(z_list[i], A, b)
        if ri==0:
            continue
        else:
            #find the first n*(n+1)/2 x-entries
            for h in range(n):      #length n
                for j in range(h,n):
                    if h==j:
                        dfx[index] += z_list[i][0]*2*ri*z_list[i][h + 1] ** 2
                    else:
                        dfx[index] += z_list[i][0]*4* ri*(z_list[i][j + 1]) * (z_list[i][h + 1])
                    index+=1

            #find the last n x-entries
            for h in range(n):
                dfx[int(n * (n + 1) / 2) + h] += z_list[i][0]*2*ri*z_list[i][h+1]
    return dfx

#Denne finner df fra gammel funksjon og legger til gradienten til logleddet
def dP(z_list, n, x, my, lambda_low, lambda_high):
    #print(x)
    function=df_model(z_list,n,x, my, lambda_low, lambda_high)
    c=c_function(x, lambda_low, lambda_high)
    dc=dc_function(x)
    for i in range(len(c)):
        function-=(my/c[i])*dc[i]
    return function

#Julie fixa noke her, antageligvis for å få identisk z-list kvar gong?
def construct_z_elliptic(n, m, A, b, area):
    #z_list = np.random.uniform(-area, area, (m, n + 1))
    #np.save("z_list",z_list) #Bruk denne først til å lagre
    z_list = np.load("z_list.npy") #Kommenter så ut de to over og hent inn med denne
    for i in range(m):
        z_list[i][0] = 1
        if compute_r_i(z_list[i], A, b) > 0:
            z_list[i][0] = -1
    return z_list