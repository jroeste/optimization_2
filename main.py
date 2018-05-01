import numpy as np
import matplotlib.pyplot as plt
import methods as meth
import functions as f
import plotting as p



'''Constants:'''
m=50
n=2
area=2.0
x_length=int(n*(n+1)/2)+n
x_initial=np.zeros(x_length)
x_initial[0], x_initial[2]=1,1
prob=0.05
min_rec,max_rec=1,4

lambda_low=0.5
lambda_high=50

if __name__ == "__main__":

    Master_Flag = {
                    0: 'Create dataset',
                    1: 'Plot'





            }[0]
    if Master_Flag =='Create dataset':
        a00 = np.random.uniform(0, 2)
        a11 = np.random.uniform(0, 2)
        minval = min(a00, a11)
        a01 = np.random.uniform(0, minval)
        #A = [[a00, a01], [a01, a11]]  # symmetric, positive definite A
        #b = np.random.uniform(-1, 1, n)
        A=[[1.5, 0.5], [0.5, 1]]
        b=[0,0]
        z_list=f.construct_z_elliptic(n, m, A, b, area)
        solution=meth.log_barrier(z_list, n, x_initial ,lambda_low, lambda_high)
        print(solution)
        A,b=f.construct_A_and_b(n, solution)
        X, Y, Z = p.make_ellipse(A, b, area, p.eval_func_model_2D)
        p.plot_dataset_2d(X, Y, Z)
        p.plot_z_points(z_list, m)
        plt.show()


    elif Master_Flag=='Plot':
        print("hei igjen")




