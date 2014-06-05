# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy
import scipy
import scipy.linalg as linalg
import matplotlib.pyplot as plt

# <markdowncell>

# Assume you have some samples with $N$ observable variables ($\vec{o}$) and you assume they are a linear mixture ($A$) of $N$ latent variables $\vec{x}$ that are drawn from independant standard normals.

# <codecell>

N = 2
M = 50000 # number of samples

# <codecell>

# emulate the latent variables that we want to discover so we can generate our observables and test
X = numpy.random.normal(size=(N,M))
plt.scatter(X[0,:], X[1,:])
None

# <codecell>

# emulate a random mixing matrix
A = numpy.random.random((2,2))
A

# <codecell>

# emulate the observable variables
O = numpy.dot(A, X)
plt.scatter(O[0,:], O[1,:])
None

# <markdowncell>

# ## The question
# 
# Can we recover $A$ and $O$, at least estimate them, by only looking at $X$?
# 
# YES!
# 
# You know that covariance of your mixed distribution is $A^2$

# <codecell>

# verify that A**2 is close to our sample covariance
samplecov = numpy.dot(O, O.T) / (M - 1)
samplecov - numpy.dot(A,A)

# <markdowncell>

# So we know can estimate $A$ by finding a $\sqrt{\text{sample covariance}}$ note, there may be more than 1 sqrt, I don't think it matters which one we find.  Since a sample covariance will always be positive semi-definite, we can do this with eigen-decomposition.

# <codecell>

def mat_sqrt(cov):
    evals,left_evecs = linalg.eig(cov, left=True, right=False)
    
    sqrtd = numpy.diag(numpy.sqrt(evals))
    return numpy.dot(numpy.dot(left_evecs, sqrtd), left_evecs.T)

# <codecell>

to_mixed_map = mat_sqrt(samplecov)

# verify it really is a sqrt
numpy.sum(samplecov - numpy.dot(to_mixed_map, to_mixed_map))

# <codecell>

to_mixed_map

# <codecell>

A

# <markdowncell>

# to_mixed_map may or may not be an estimate of $A$ proper, depending on if we found $A$ in our $\sqrt{}$ operation or some other root.  In the end it doesn't matter because it will map from *some* standard normal to our distribution.

# <codecell>

tmp = numpy.dot(to_mixed_map, X)
plt.scatter(tmp[0,:], tmp[1,:])
None

# <markdowncell>

# So it's the correct shape.

# <markdowncell>

# # What about finding reasonable values of our latent variables $\vec{x}$?
# 
# Well, $A^{-1}$ maps from our observed distribution to the standard normal

# <codecell>

to_unmxed_map = linalg.inv(to_mixed_map)
to_unmxed_map

# <codecell>

# verify that it looks like it was pulled from a standard normal
tmp = numpy.dot(to_unmxed_map, O)
plt.scatter(tmp[0,:],tmp[1,:])
None

# <markdowncell>

# ## What if our sample matrix is not invertable?
# 
# If the sample covariance is not full rank, then we can calculate the inversion along k principle axises for any $k$ we wish and ignore the rest of the axises.  For large matrices you would specify k and use sparse libraries, but we'll demonstrate using all non-zero principle axises

# <codecell>

def estimate_invert(X):
    evals, left_evecs = linalg.eig(X, left=True, right=False)
    evals[evals < 0.00001] = 0.0 # zero out values that are probably rounding errors
    
    # remove zero axises
    idxs = evals != 0
    evals = evals[idxs]
    left_evecs = left_evecs[:,idxs]
    
    invd = numpy.diag(1.0 / evals)
    return numpy.dot(left_evecs, numpy.dot(invd, left_evecs.T))

# <markdowncell>

# # PCA does the same thing
# 
# More details to follow when I have a few minutes.

