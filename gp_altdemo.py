# This demonstration gives a slightly different view of the two stage sampling
# process demonstrated in gp_minimal.py. See that file for more details. Here we
# reinforce that sampling from the posterior really is just continuing the prior
# sampling process given the values that we've seen. I like this version of the
# demo because there's less linear algebra than in gp_minimal.py, but this
# presentation is less standard.
#
# Iain Murray, November 2016

import numpy as np
import matplotlib.pyplot as plt


## The kernel function (as in gp_minimal.py)
######################################################################

rbf_fn = lambda X1, X2: \
        np.exp((np.dot(X1,(2*X2.T))-np.sum(X1*X1,1)[:,None]) - np.sum(X2*X2,1)[None,:])
gauss_kernel_fn = lambda X1, X2, ell, sigma_f: \
        sigma_f**2 * rbf_fn(X1/(np.sqrt(2)*ell), X2/(np.sqrt(2)*ell))
k_fn = lambda X1, X2: gauss_kernel_fn(X1, X2, 3.0, 10.0)


## Sampling from the prior
######################################################################

# Pick the input locations that we want to see the function at.
X_train = np.array([2,4,6,8])[:,None] + 0.01
X_test = np.arange(0, 10, 0.02)[:,None]
X_all = np.vstack([X_train, X_test])
N_train = X_train.shape[0]
N_all = X_all.shape[0]

# The joint distribution over function values has zero mean and covariance
# K_all = np.dot(L_all, L_all.T)
K_all = k_fn(X_all, X_all) + 1e-9*np.eye(N_all)
L_all = np.linalg.cholesky(K_all)

# Function values can be sampled with: L_all*nu, where nu = randn(N_all).
# Because L_all is lower-triangular, the first N_train function values depend
# only on the first N_train values of nu. We pick those first:
nu1 = np.random.randn(N_train)
plt.figure(1)
plt.clf()
for ii in range(3):
    # Then we consider different samples from the prior that complete those
    # first N_train values in different ways:
    nu2 = np.random.randn(N_all - N_train)
    nu = np.hstack([nu1, nu2])
    f_all = np.dot(L_all, nu)
    # These x's will fall on top of each other for each loop, as nu1 is shared:
    plt.plot(X_train, f_all[:N_train], 'x', markersize=20, markeredgewidth=2)
    # But we'll get different completions for different nu2. These are
    # samples from the posterior given the 'x' observations.
    plt.plot(X_test, f_all[N_train:], '-', linewidth=2)

plt.legend(['train points', 'completions / posterior samples'])
plt.xlabel('x')
plt.ylabel('f')
plt.show()


# Want to see samples from the posterior given noisy observations? You could
# insert the following two lines beneath the definition of K_all:
#noise_var = 1.0
#K_all[:N_train, :N_train] = K_all[:N_train, :N_train] + noise_var*np.eye(N_train)

# You could extend the demo to plot mean and error bars like in gp_minimal.py

# Of course we don't see the random numbers nu1 directly when we observe data.
# However, they are known: we can solve for nu1 from the observed values.
# I should use a specialist triangular solver, but just an illustration:
nu1_from_obs = np.linalg.solve(L_all[:N_train, :N_train], f_all[:N_train])
assert(np.max(np.abs(nu1_from_obs - nu1)) < 1e-9)


# Notice how almost all of the code above is comments, plotting, and tracking
# which data points are which. Little maths is required to sample realizations
# of complex functions given data.

