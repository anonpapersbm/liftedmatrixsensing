import jax
import jax.numpy as jnp
import optax
import numpy as np
from scipy import linalg as LA
from sklearn.cluster import KMeans

from itertools import chain, repeat, product


def prunerank(U, A):
    n = U.shape[0]
    try:
        u = U.reshape((n,))
        return u, A
    except:
        u = jnp.matmul(U, U.T).reshape((n*n,))
        m = A.shape[0]
        A = jnp.outer(A, A).reshape((m*m, n*n, n*n))
        return u, A

def heaprank(u, r):
    if r > 1:
        n = int(jnp.sqrt(u.shape[0]))
        u = u.reshape(n,n)
        U, Sigma, _ = jnp.linalg.svd(u)
        w = jnp.matmul(U[:,:r], jnp.diag(jnp.sqrt(Sigma[:r])))
        return w.reshape(n, r)
    else:
        return u.reshape((-1,))

def get_AA_idxs(lvl):
    '''
    tests:
    get_AA_idxs(0) == [(0, 1, 2), (0, 3, 4)]
    get_AA_idxs(1) == [(0, 1, 2), (0, 3, 4), (5, 6, 7), (5, 8, 9)]
    '''
    AA_idxs = [[tuple(np.asarray([i, i+1, i+2]) +i*4), tuple(np.asarray([i, i+3, i+4])+i*4)] for i in range(lvl+1)]
    AA_idxs = list(chain(*AA_idxs))
    return AA_idxs

def get_ww_idxs(lvl):
    '''
    tests:
    get_ww_idxs(0) == [(1), (2), (3), (4)]
    get_ww_idxs(1) == [(1, 6), (2, 7), (3, 8), (4, 9)]
    '''
    ww_idxs = [tuple(np.asarray(list(range(lvl+1)))*5 + i+1)  for i in range(4)]
    return ww_idxs

def get_w_z_idxs(lvl):
    '''
    tests:
    get_w_z_idxs(0) == ([(1,), (2,)], [(3,), (4,)])
    get_w_z_idxs(1) == ([(1, 6,), (2, 7,)], [(3,), (8,), (4,), (9,)])
    '''
    w_idxs = [tuple(np.asarray(list(range(lvl+1)))*5 + i+1) for i in range(2)]
    z_idxs = [tuple(np.asarray(list(range(lvl+1)))*5 + i+1) for i in [2,3]]
    # z_idxs = [(3+5*l,) for l in range(lvl+1)] + [(4+5*l,) for l in range(lvl+1)]
    return w_idxs, z_idxs

def get_A_idxs(lvl):
    '''
    tests:
    get_A_idxs(0) == [(0, 1, 2)]
    get_A_idxs(1) == [(0, 1, 2), (3, 4, 5)]
    '''
    A_idxs = [tuple(np.asarray([i, i+1, i+2]) +i*2) for i in range(lvl+1)]
    return A_idxs


def idxs_to_args(tensor, idxs):
    return list(chain(*list(zip(repeat(tensor), idxs))))


def elevate_initialization(w_in, level,flatten=False):
    '''
    w_in.shape = (n,)
    w_out.shape = (n, .. level+1 times .., n)
    '''
    einsum_w_args = list(chain(*[(w_in, (i,)) for i in range(level+1)]))
    w_norank = jnp.einsum(*einsum_w_args)
    if flatten:
        w_norank = w_norank.reshape(jnp.prod(jnp.asarray(w_norank.shape)))
    return w_norank


def elevate_A(A, level,flatten=False):
    einsum_A_args = list(chain(*[(A, (i*3,i*3+1,i*3+2)) for i in range(level+1)]))
    A_lifted = jnp.einsum(*einsum_A_args)

    return A_lifted

def data_loss_new(w, z_lifted, A_lifted,lvl):
    '''w.shape = (n, .. level+1 times .., n)
    We want to take in flattened w, since we want hessian as matrix, but z and A can be tensors
    '''
    n = z_lifted.shape[0]
    w = jnp.reshape(w,tuple(n for i in range(lvl+1)))

    idx_A = [i for i in range(3*(lvl+1))]
    idx_1 = [1+3*i for i in range(lvl+1)]
    idx_2 = [2+3*i for i in range(lvl+1)]
    
    Aww = jnp.einsum(A_lifted,idx_A,w,idx_1,w,idx_2)

    Azz = jnp.einsum(A_lifted,idx_A,z_lifted,idx_1,z_lifted,idx_2)

    diff = jnp.reshape(Aww-Azz,-1)

    loss = 0.5*jnp.dot(diff,diff)

    return loss


def data_loss_z(w, z_lifted, A,calc_grad=False):
    '''w.shape = (n, .. level+1 times .., n)'''
    lvl = len(w.shape)-1

    AA_idxs = get_AA_idxs(lvl)
    AA_args = idxs_to_args(A, AA_idxs)

    ww_idxs = get_ww_idxs(lvl)
    ww_args = idxs_to_args(w, ww_idxs)
    Hww_args = AA_args + ww_args

    w_idxs, z_idxs = get_w_z_idxs(lvl)
    wz_args = idxs_to_args(w, w_idxs) + idxs_to_args(z_lifted, z_idxs)
    Hwz_args = AA_args + wz_args

    #too speed up computation by neglecting constant terms
    if calc_grad==False:
        zz_args = idxs_to_args(z_lifted, ww_idxs)
        Hzz_args = AA_args + zz_args

        error_data = jnp.einsum(*Hww_args) + jnp.einsum(*Hzz_args) - 2 * jnp.einsum(*Hwz_args)
    else:
        error_data = jnp.einsum(*Hww_args) - 2 * jnp.einsum(*Hwz_args)

    return 0.5*error_data


def adam_optimize(problem,
          lr,
          epochs,
          gradnorm_epsilon,
          loss_epsilon = float('-inf')
          ):
    f, w, params = problem
    optimizer = optax.adam(lr)
    # Obtain the `opt_state` that contains statistics for the optimizer.
    opt_state = optimizer.init(w)

    l_g_fn = jax.value_and_grad(f)

    # the main training loop
    for _ in range(epochs):
        loss, grads = l_g_fn(w, *params)
        grads_flatten, _ = jax.flatten_util.ravel_pytree(grads)
        if loss < loss_epsilon or jnp.linalg.norm(grads_flatten) < gradnorm_epsilon:
            break
        updates, opt_state = optimizer.update(grads, opt_state)
        w = optax.apply_updates(w, updates)

    return loss, grads, w
    
def tensor_PCA(tensor,
                        lr, #=0.05,
                        epochs, #=2000,
                        gradnorm_epsilon,
                        lambd_v=None, key=None):
    # either v or key must be not None

    def loss(eigenval_eigenvec, tensor):
        lambd, v = eigenval_eigenvec
        k = len(tensor.shape)
        for _ in range(len(tensor.shape)):
            tensor = jnp.inner(tensor, v)
        first_term = jnp.square(lambd)*jnp.power(jnp.linalg.norm(v), 2*k)
        res = first_term - 2*lambd*tensor
        return res

    s = tensor.shape[0]
    if lambd_v is None:
        key1, key2 = jax.random.split(key)
        v = jax.random.normal(key1, shape=(s,))/jnp.sqrt(s)
        lambd = jax.random.normal(key2, shape=())
    else:
        lambd, v = lambd_v

    loss, grads, lambd_v = adam_optimize((loss, (lambd, v), (tensor)),
                  lr=lr,
                    epochs=epochs,
                    gradnorm_epsilon=gradnorm_epsilon)
    lambd, v = lambd_v

    return jnp.power(lambd, 1/len(tensor.shape))*v

def drop(w,
         lr=0.5,
         epochs=2000,
         gradnorm_epsilon=1e-6, key=jax.random.PRNGKey(0)):
    n = w.shape[0]
    w = tensor_PCA(w,
                            lr=lr,
                            epochs=epochs,
                            gradnorm_epsilon=gradnorm_epsilon,
                            key=key)
    return w.reshape(n,)


def gd_optimize(problem,
          lr,
          epochs,
          gradnorm_epsilon,
          loss_epsilon = float('-inf')
          ):
    f, w,z,A = problem

    level = len(z.shape)-1

    l_g_fn = jax.value_and_grad(f) #by default takes derivative wrt the first variable

    # the main training loop
    for _ in range(epochs):
        loss, grads = l_g_fn(w,z,A,level)
        grads_flatten, _ = jax.flatten_util.ravel_pytree(grads)
        if loss < loss_epsilon or jnp.linalg.norm(grads_flatten) < gradnorm_epsilon:
            break
        w = w - lr*grads

    return loss, grads, w


def adam_optimize_traj(problem,
          lr,
          epochs,
          gradnorm_epsilon,
          loss_epsilon = float('-inf'), trajectory_params = (10,3,1)):
    f, w, params = problem

    (record_interval,n,level) = trajectory_params
    trajectory = np.zeros((n,np.floor(epochs/record_interval).astype(int)))

    optimizer = optax.adam(lr)
    # Obtain the `opt_state` that contains statistics for the optimizer.
    opt_state = optimizer.init(w)

    l_g_fn = jax.value_and_grad(f)

    # the main training loop
    for epoch in range(epochs):
        loss, grads = l_g_fn(w, *params)
        grads_flatten, _ = jax.flatten_util.ravel_pytree(grads)
        if loss < loss_epsilon or jnp.linalg.norm(grads_flatten) < gradnorm_epsilon:
            break
        updates, opt_state = optimizer.update(grads, opt_state)
        w = optax.apply_updates(w, updates)
        if np.mod(epoch,record_interval) == 0:
            w_recover = drop(jnp.reshape(w,tuple(n for i in range(level+1)))) 
            if np.isnan(w_recover).any():
                trajectory[:,int(epoch/record_interval)] = trajectory[:,int(epoch/record_interval)-1]
            else:
                trajectory[:,int(epoch/record_interval)] = w_recover

    return loss, grads, w, trajectory #returns the flattened w

def solve(z, z_lifted, A_lifted, level, key, init_mag,
          w_0=None,
          lr=0.02,
          epochs=2000,
          loss_epsilon=1e-6,
          gradnorm_epsilon=1e-5,
          drop_lr=0.5,
          drop_epochs=2000,
          drop_gradnorm_epsilon=1e-6):
    if len(z.shape) == 1:
        n = z.shape[0]
        r = 1
    else:
        n, r = z.shape

    lift_key, drop_key = jax.random.split(key)
    if w_0 is not None:
        if len(w_0.shape) < level+1:
            w = elevate_initialization(w_0, level,flatten=True)
    else:
        w = jax.random.normal(lift_key, shape=z.shape*(level+1))
        w = w/jnp.sqrt(jnp.prod(jnp.asarray(w.shape)))
        w = w.reshape(-1)*init_mag #flattens w
    # gd_jit = jax.jit(gd_optimize)

    # loss, grads, w = gd_optimize((data_loss_new, w, z_lifted ,A_lifted), lr=lr, epochs=epochs,
    #               loss_epsilon=loss_epsilon,
    #               gradnorm_epsilon=gradnorm_epsilon)

    loss, grads, w = adam_optimize((data_loss_new, w, (z_lifted ,A_lifted,level)), lr=lr, epochs=epochs,
                  loss_epsilon=loss_epsilon,
                  gradnorm_epsilon=gradnorm_epsilon)


    return w


def solve_trajectory(z, z_lifted, A_lifted, level, key, init_mag,
          w_0=None,
          lr=0.02,
          epochs=2000,
          loss_epsilon=1e-6,
          gradnorm_epsilon=1e-5,
          drop_lr=0.5,
          drop_epochs=2000,
          drop_gradnorm_epsilon=1e-6):
    if len(z.shape) == 1:
        n = z.shape[0]
        r = 1
    else:
        n, r = z.shape

    lift_key, drop_key = jax.random.split(key)
    if w_0 is not None:
        if len(w_0.shape) < level+1:
            w = elevate_initialization(w_0, level,flatten=True)
    else:
        w = jax.random.normal(lift_key, shape=z.shape*(level+1))
        w = w/jnp.sqrt(jnp.prod(jnp.asarray(w.shape)))
        w = w.reshape(-1)*init_mag
    # gd_jit = jax.jit(gd_optimize)

    loss, grads, w, trajectory = adam_optimize_traj((data_loss_new, w, (z_lifted ,A_lifted,level)), lr=lr, epochs=epochs,
                  loss_epsilon=loss_epsilon,
                  gradnorm_epsilon=gradnorm_epsilon,trajectory_params=(20,n,level))

    return w, trajectory


def output_A_MS(mask):
    #outputs a n*n, n,n tensor
    n = mask.shape[0]
    A = np.zeros((n**2,n,n))
    for [i,j] in product(np.arange(n),repeat=2):
        temp = np.zeros((n,n))
        temp[i,j]=mask[i,j]
        temp = (temp+temp.T)/2
        A[i*n+j,:,:]=temp

    return A

def get_grad(problem,x):
    loss_fn, z_lifted,A,level = problem
    grad_fn = jax.jacrev(loss_fn)
    grad = grad_fn(x,z_lifted,A,level)
    grad_flatten, _ = jax.flatten_util.ravel_pytree(grad)
    return (grad,jnp.linalg.norm(grad_flatten))


def hessian(f):
    return jax.jacfwd(jax.jacrev(f))


def hessian_smallest_eigval(problem, solution):
    # returns the smallest eigenvalu of the Hessian matrix at the point w in
    # the problem given with (y, mask, rank)
    loss_fn, z_lifted,A,level = problem
    H = hessian(loss_fn)(solution, z_lifted,A,level)
    # Hmat = H.reshape((jnp.prod(jnp.asarray(solution.shape)), jnp.prod(jnp.asarray(solution.shape))))
    eigenvals, _ = jnp.linalg.eigh(H)
    return float(min(eigenvals).real)