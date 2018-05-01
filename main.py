import numpy as np
import matplotlib.pyplot as plt
import methods as meth
import functions as f
import plotting as p



'''Constants:'''
m=500
n=2
my=10
area=2.0
x_length=int(n*(n+1)/2)+n
x_initial=np.zeros(x_length)
x_initial[0], x_initial[2]=1,1
prob=0.05
min_rec,max_rec=1,4


lambda_low=0.1
lambda_high=50

if __name__ == "__main__":

    Master_Flag = {
                    0: 'Create dataset',
                    1: 'Plot',
                    2: 'Testing functions',




            }[0]
    if Master_Flag =='Create dataset':
        A=[[1, 0.5], [0.5, 0.5]]    #Denne lager klassifiseringsellipse. [[1, 0.5], [0.5, 0.5]] er eksempel på
                                    # testproblem der løsningen er grei, f eks [[0.001, 3], [3, 0.005]] (og lignende)
                                    #er målet å få til å funke
        b=[0,0]
        z_list=f.construct_z_elliptic(n, m, A, b, area)     #Lager z-liste fra ellipsen over. Denne lager tilfeldige punkt, som er litt uheldig for testing.
        solution=meth.log_barrier(z_list, n, x_initial ,lambda_low, lambda_high) #Bruker log-barrier for å finne løsning

        # Resten plotter
        A,b=f.construct_A_and_b(n, solution)
        X, Y, Z = p.make_ellipse(A, b, area, p.eval_func_model_2D)
        p.plot_dataset_2d(X, Y, Z)
        p.plot_z_points(z_list, m)
        plt.show()


    elif Master_Flag=='Plot':
        print("hei igjen")

    elif Master_Flag=='Testing functions':
        x=np.linspace(-5,5,m)




