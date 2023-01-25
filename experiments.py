import jax
import jax.numpy as jnp
import optax
import numpy as np
from scipy import linalg as LA
from sklearn.cluster import KMeans

from itertools import chain, repeat, product

from functions import *

%matplotlib widget
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dist_to_gt_list = []
success_rate_list = []
for (n,level) in product(np.array([3,4,5]),np.array([2])):
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

    trial_num = 40

    all_trajectories = []
    all_trajectories_lifted = []

    dist_to_gt = np.zeros((2,trial_num))

    for trial in range(trial_num):
        init_mag = 0.01

        ##############################
        #unlifted problem
        w_final= solve(z, z, A, 0, jax.random.PRNGKey(trial), init_mag)
        dist_to_gt[0,trial] = min(LA.norm(w_final-z),LA.norm(w_final+z))
        #############################
        #lifted problem
        w_final_lifted= solve(z, z_lifted, A_lifted, level, jax.random.PRNGKey(trial), init_mag)
        dist_to_gt[1,trial] = min(LA.norm(w_final_lifted-elevate_initialization(z,level,flatten=True)),LA.norm(w_final_lifted-elevate_initialization(-z,level,flatten=True)))
    
    dist_to_gt_list.append(dist_to_gt)
    rate1 = len(dist_to_gt[0,dist_to_gt[0,:] <= 0.1])/len(dist_to_gt[0,:])
    rate2 = len(dist_to_gt[1,dist_to_gt[1,:] <= 0.1])/len(dist_to_gt[1,:])
    success_rate_list.append((rate1,rate2))

print(success_rate_list)

ground_truth = np.array([[1,0,1],[-1,0,-1]])
local_min = np.array([[-1,0,1],[1,0,-1]])
    
fig = plt.figure()
# ax = Axes3D(fig)
ax = fig.add_subplot(111, projection = '3d')
# for trajectory in all_trajectories:
#     # ax.plot(points[0], points[1], points[2], marker = 'x')
#     ax.plot(trajectory[0], trajectory[1], trajectory[2])
#     #ax.scatter(*trajectory[np.newaxis,:,0].T,marker='v')
for trajectory in all_trajectories_lifted:
    ax.plot(trajectory[0], trajectory[1], trajectory[2])
    # ax.scatter(*trajectory[np.newaxis,:,0].T,marker='v')
ax.scatter(*ground_truth.T,marker = 'o',color = 'red',s=100)
ax.scatter(*local_min.T,marker = 'x',color = 'black',s=100)
#ax.axes.set_ylim3d(bottom=-0.1, top=0.1) 
plt.show()