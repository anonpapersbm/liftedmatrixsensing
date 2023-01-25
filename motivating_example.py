import jax
import jax.numpy as jnp
import optax
import numpy as np
from scipy import linalg as LA
from sklearn.cluster import KMeans

from itertools import chain, repeat, product

from functions import *

level=2
n=5
only_lifted=True

#for specific dimension
z = np.zeros(n)
for k in range(int(np.ceil(n/2))):
    z[2*k]=1

M_star = np.outer(z,z)

eps = 0.3

mask = np.ones((n,n))*eps
for i in range(n):
    mask[i,i]=1
    for k in (np.arange(int(np.floor(n/2)))+1):
        mask[i,2*k-1]=1
        mask[2*k-1,i]=1

A = output_A_MS(mask)
b = np.einsum(A,[0,1,2],M_star,[1,2])

z_lifted = elevate_initialization(z,level)
A_lifted = elevate_A(A,level)

trial_num = 20

terminal_X = np.zeros((n,trial_num))
# terminal_X_lifted = np.zeros((n,)*(level+1)+(trial_num,))
terminal_X_lifted = np.zeros((n**(level+1),trial_num))


for trial in range(trial_num):
    init_mag = 0.1

    ##############################
    #unlifted problem
    terminal_X[:,trial] = solve(z, z, A, 0, jax.random.PRNGKey(trial), init_mag)

    #############################
    #lifted problem
    terminal_X_lifted[...,trial] = solve(z, z_lifted, A_lifted, level, jax.random.PRNGKey(trial), init_mag)

min_eigvals = np.zeros((2,int(np.power(2,np.ceil(n/2)))))
grad_mags = np.zeros((2,int(np.power(2,np.ceil(n/2)))))

# find the 2^(n/2) SOCP in the unlifted problem via Kmeans
data =  np.swapaxes(terminal_X[:,:],0,1)
kmeans = KMeans(n_clusters=int(np.power(2,np.ceil(n/2))))
kmeans.fit(data)

for i in range(kmeans.cluster_centers_.shape[0]):
    x = kmeans.cluster_centers_[i].reshape(n)
    min_eigvals[0,i] = hessian_smallest_eigval((data_loss_new, z,A,0), x)
    _, grad_mag = get_grad((data_loss_new, z,A,0), x)
    grad_mags[0,i] = grad_mag

    #for lifted
    w = elevate_initialization(x, level,flatten=True)
    min_eigvals[1,i] = hessian_smallest_eigval((data_loss_new, z_lifted,A_lifted,level), w)
    _, grad_mag =get_grad((data_loss_new, z_lifted,A_lifted,level), w)
    grad_mags[1,i] = grad_mag

print(min_eigvals)