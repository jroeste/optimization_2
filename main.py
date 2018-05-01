import numpy as np
import matplotlib.pyplot as plt
import methods as m
import functions as f
import plotting as p



'''Constants:'''
m=50
n=2
my=10
area=2.0
x_length=int(n*(n+1)/2)+n
prob=0.05
min_rec,max_rec=1,4

lambda_low=1e-2
lambda_high=1e2

if __name__ == "__main__":

    Master_Flag = {
                    0: 'Create dataset',
                    1: 'Plot',
                    2: 'Testing functions',




            }[2]
    if Master_Flag =='Create dataset':
        print("Creates dataset")


    elif Master_Flag=='Plot':
        print("hei igjen")

    elif Master_Flag=='Testing functions':
        x=np.linspace(-5,5,m)




