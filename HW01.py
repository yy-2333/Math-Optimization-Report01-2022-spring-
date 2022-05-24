from cProfile import label
import numpy as np
import matplotlib.pyplot as plt


A = np.array([[-11.24141852, -30.27686549, 24.41045864, -32.48539868, 28.77265676],
 [  7.66104597, 32.74804359, -15.04176741, -7.92810814, 20.59294248],
 [-12.06767453, -3.07406867, 4.71301872, -6.69717506, 26.52537247]])
A_T = A.T
b = np.array([-16.05858601, 11.5077113, 25.5509359])
lambd = 1
ident = np.eye(5)
L = 2*max(np.linalg.eigvals(A_T @ A + lambd*ident))
step = 1 / L

def target_function(w):
    Aw = A @ w
    return (np.linalg.norm(b-Aw))**2 + lambd*((np.linalg.norm(w))**2)

def gradient(w):
    return 2*(A_T @ (A @ w - b)) + 2*lambd*w

x = np.ones(5)
eps = 0.01
max_iter = 500

iter_list_0= []
value_list_0 = []

iter_list = []
value_list = []

iter_list_10 = []
value_list_10 = []

iter_list_Ar = []
value_list_Ar = []

iter_list_Ne = []
value_list_Ne = []


def GD1(target_fn,gradient_fn,initial,eps,maxerr):
    iter = 0
    value = initial
    iter_list.append(iter)
    value_list.append(target_fn(value))
    while True and (iter < maxerr):
        iter += 1
        grad = gradient_fn(value)
        next_value = value - step*grad
        err = np.linalg.norm(next_value - value)
        if  err < eps:
            iter_list.append(iter)
            value_list.append(target_fn(next_value))
            print('iter:',iter)
            print(target_fn(next_value))
            print('moving distance:',err)
            break
        else:
            iter_list.append(iter)
            value_list.append(target_fn(next_value))
            print('iter:',iter)
            print('function value:',target_fn(next_value))
            print('moving distance:',err)
            value = next_value

def GD0(target_fn,gradient_fn,initial,eps,maxerr):
    iter = 0
    value = initial
    iter_list_0.append(iter)
    value_list_0.append(target_fn(value))
    while True and (iter < maxerr):
        iter += 1
        grad = gradient_fn(value)
        next_value = value - step*grad
        err = np.linalg.norm(next_value - value)
        if  err < eps:
            iter_list_0.append(iter)
            value_list_0.append(target_fn(next_value))
            print('iter:',iter)
            print(target_fn(next_value))
            print('moving distance:',err)
            break
        else:
            iter_list_0.append(iter)
            value_list_0.append(target_fn(next_value))
            print('iter:',iter)
            print('function value:',target_fn(next_value))
            print('moving distance:',err)
            value = next_value    

def GD10(target_fn,gradient_fn,initial,eps,maxerr):
    iter = 0
    value = initial
    iter_list_10.append(iter)
    value_list_10.append(target_fn(value))
    while True and (iter < maxerr):
        iter += 1
        grad = gradient_fn(value)
        next_value = value - step*grad
        err = np.linalg.norm(next_value - value)
        if  err < eps:
            iter_list_10.append(iter)
            value_list_10.append(target_fn(next_value))
            print('iter:',iter)
            print(target_fn(next_value))
            print('moving distance:',err)
            break
        else:
            iter_list_10.append(iter)
            value_list_10.append(target_fn(next_value))
            print('iter:',iter)
            print('function value:',target_fn(next_value))
            print('moving distance:',err)
            value = next_value   

def GD_Armijo(target_fn,gradient_fn,initial,eps,maxerr):
    iter = 0
    value = initial
    iter_list_Ar.append(iter)
    value_list_Ar.append(target_fn(value))
    while True and (iter < maxerr):
        iter += 1
        grad = gradient_fn(value)
        stepsize = 1
        alpha = 0.4
        beta = 0.8
        while target_fn(value-stepsize*grad) > target_fn(value)-alpha*stepsize*(grad @ grad):
            stepsize *= beta
        next_value = value - stepsize*grad
        err = np.linalg.norm(next_value - value)
        if  err < eps:
            iter_list_Ar.append(iter)
            value_list_Ar.append(target_fn(next_value))
            print('iter:',iter)
            print(target_fn(next_value))
            print('moving distance:',err)
            break
        else:
            iter_list_Ar.append(iter)
            value_list_Ar.append(target_fn(next_value))
            print('iter:',iter)
            print('function value:',target_fn(next_value))
            print('moving distance:',err)
            value = next_value
            

def Nesterov_acceleration(target_fn,gradient_fn,initial,eps,maxerr):
    iter = 0
    value = initial
    stepsize = 1
    alpha_a = 0.4
    beta_a = 0.8
    iter_list_Ne.append(iter)
    value_list_Ne.append(target_fn(value))
    iter += 1
    grad = gradient_fn(value)
    while target_fn(value-stepsize*grad) > target_fn(value)-alpha_a*stepsize*(grad @ grad):
        stepsize *= beta_a
    next_value = value - stepsize*grad
    iter_list_Ne.append(iter)
    value_list_Ne.append(target_fn(next_value))
    alpha = step
    while True and (iter < maxerr):
        iter += 1
        beta = (iter-1)/(iter+3)

        next_next_value = next_value  + beta*(next_value-value) - alpha * gradient_fn(next_value+beta*(next_value-value))
        err = np.linalg.norm(next_next_value - next_value)
        if  err < eps:
            iter_list_Ne.append(iter)
            value_list_Ne.append(target_fn(next_next_value))
            print('iter:',iter)
            print(target_fn(next_next_value))
            print('moving distance:',err)
            break
        else:
            iter_list_Ne.append(iter)
            value_list_Ne.append(target_fn(next_next_value))
            print('iter:',iter)
            print('function value:',target_fn(next_next_value))
            print('moving distance:',err)
            value = next_value
            next_value = next_next_value
            

GD1(target_function,gradient,np.ones(5).T,eps,max_iter)
#lambd = 0
#GD0(target_function,gradient,np.ones(5).T,eps,max_iter)
#lambd = 10
#GD10(target_function,gradient,np.ones(5).T,eps,max_iter)
#GD_Armijo(target_function,gradient,np.ones(5).T,eps,max_iter)
Nesterov_acceleration(target_function,gradient,np.ones(5).T,eps,max_iter)
#print(A)
#print(b)

plt.plot(iter_list,value_list,label='Constant step-size')
plt.plot(iter_list_Ne,value_list_Ne,label='Nesterov’s acceleration')
#plt.plot(iter_list_Ar,value_list_Ar,label='With backtracking')
#plt.plot(iter_list_0,value_list_0,label='lambda=0')
#plt.plot(iter_list_10,value_list_10,label='lambda=10')
plt.xlabel('Iteration number')
plt.ylabel('Function value')
plt.legend(loc='upper left')
#plt.plot(iter_list_10,value_list_10)




#


plt.show()
