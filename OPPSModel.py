import math
import numpy as np
import random
# define the objective function as a function of n & m not T.
# first need to define cn and xn
n=50
m=10
w=0
# t=50
searchSpace=np.array([[w,w,w,w,w,w,w,3,6,w,2,w,1,w,w,6,1,w,1,7,6,8,1,w,4,7,w,1,6,w,8,w,w,w,4,5,7,8,4,2,8,w,3,w,w,w,w,w,w,w],
[w,w,w,w,w,6,8,1,w,4,7,w,1,6,w,2,w,7,8,w,4,6,w,1,4,8,w,7,5,8,w,1,8,w,3,w,6,7,1,8,5,w,1,5,w,w,w,w,w,w],
[w,w,w,w,w,4,6,w,1,4,8,w,7,5,8,w,7,w,w,8,w,5,5,w,w,3,4,w,1,5,6,w,w,6,8,w,4,w,w,w,3,2,w,w,w,w,w,w,w,w],
[w,w,w,w,w,w,5,5,w,w,3,4,w,1,5,6,8,7,6,8,8,2,3,1,w,2,w,2,1,4,w,8,7,4,8,w,1,2,w,w,w,w,w,w,w,w,w,w,w,w],
[w,w,w,w,w,8,2,3,1,w,2,w,2,1,4,w,3,4,w,7,5,w,1,8,5,w,6,6,w,w,5,7,w,2,4,w,w,w,4,6,6,w,7,w,w,w,w,w,w,w],
[w,w,w,w,w,5,w,1,8,5,w,6,6,w,w,5,2,w,w,3,3,8,w,4,2,6,6,4,w,7,2,w,w,w,w,w,5,7,3,2,2,w,5,6,w,w,w,w,w,w],
[w,w,w,w,w,3,8,w,4,2,6,6,4,w,7,2,w,7,8,2,w,5,5,w,w,w,w,3,w,4,7,w,2,1,1,6,w,w,w,w,1,8,w,6,w,w,w,w,w,w],
[w,w,w,w,w,w,5,5,w,w,w,w,3,w,4,7,w,w,5,8,w,2,4,8,3,1,w,1,3,3,7,w,w,w,7,4,3,4,6,7,1,4,w,3,w,w,w,w,w,w],
[w,w,w,w,w,w,2,4,8,3,1,w,1,3,3,7,w,w,w,4,w,w,1,6,w,w,w,w,w,w,5,w,w,4,4,w,1,2,6,w,5,5,1,w,w,w,w,w,w,w],
[w,w,w,w,w,8,w,1,6,w,w,w,w,w,w,5,w,4,4,5,w,w,w,3,6,w,2,w,1,w,w,2,w,w,1,5,8,2,2,7,6,w,w,8,w,w,w,w,w,w]])
# c=np.zeros((m,n))
# x=np.zeros((m,n))
# product=c*x
# productFilled=product
# o=0

# # o is the number of the ore blocks
# objectiveFunction=np.sum(product)
# # constraints
# # Mining Capavity Constraint, wn for each block deemed to be 0.5 tone as average
# # ore block number is equal to variable o.
wn=np.zeros((n,1))
for i in range(n):
    # for j in range(n):
        wn[i]=0.5
# xmo=np.zeros((m,o))
# # constraint of the mining capacity
# np.sum(np.matmul(x,wn))<=10
# np.sum(np.matmul(x,wn))>=6
# # sequencing consttraints

def sequencingConstraints(x:np.array)->bool:
    L=3
    t1=0
    m,n=x.shape
    for i in range(1, m-1):
        for j in range(n):
            if x[i, j] <= x[i-1, j-1] + x[i, j-1] + x[i+1, j-1]:
                return False
    return True
# indexSpace=np.zeros(np.size(searchSpace))
# # produce the mask matrix for the search space
# for i in range(len(searchSpace)):
#     for j in range(len(searchSpace[i])):
#         indexSpace[i][j]='ij'
''' terminals for the tree that includes at least # of ore  and atmost nxm blocks'''
'''making the initial tree for the GP ... '''
# search space of open pit plan with 0 for w and 1 for the ore
# initial tree for defining the individual
# x0=np.random.choice([0,1], size=(m,n))
# x0=searchSpace

# randomSelection=random.randint(o,m*n)
# Flatten the matrix to a 1D array and randomly select 'n' elements
# selectedIndices = np.random.choice(len(c0.flatten()), size=randomSelection, replace=False)
# c= [[c0.flatten()[i],i] for i in selectedIndices]
# x= [int(bool(csmall)) for csmall in c]
# parameterize the objective function
def block_value(x:np.array)->np.array:
    o=0
    c=np.zeros((m,n))
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j]==1:
                c[i][j]=80.8
                o+=1
            elif x[i][j]==2:
                c[i][j]=24.26
                o+=1
            elif x[i][j]==3:
                c[i][j]=13.14
                o+=1
            elif x[i][j]==4:
                c[i][j]=15.84
                o+=1
            elif x[i][j]==5:
                c[i][j]=6.54
                o+=1
            else:
                c[i][j]=0.0
    return c

def generate_initial_chromosome():
    # h=0
    # while(h<=(m*n)):
    x=np.zeros((m,n),dtype=bool)
    c0=np.zeros((m,n),dtype=float)
    x0=np.random.choice(searchSpace.flatten(), size=(m,n)).astype(bool)
    c0=block_value(x0)
    # constraint of the mining capacity
    # if np.sum(np.matmul(x0,wn))<=10:
    #     if np.sum(np.matmul(x0,wn))>=6:
    #         if sequencingConstraints(x0):
    #             break
    # h+=1
    # x=[[int(bool(i)) for i in range(len(x0[0]))] for j in range(len(x0))]
    for i in range(len(x0)):
        for j in range(len(x0[i])):
            x[i][j]=x0[i][j]
    return x,c0

# for i in generate_initial_chromosome()[0]:
#         for smalli in i:
    
#             print(type(smalli))
# print(generate_initial_chromosome())
print(block_value((searchSpace)))














