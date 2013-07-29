from scipy import sparse
import numpy as np

def inner(A, B):
    if sparse.issparse(A) and sparse.issparse(B):
        return (A*B.T)[0,0]
    if not sparse.issparse(A) and not sparse.issparse(B):
        return np.inner(A, B)
    else:
        raise ValueError('sparsity of arguments is not consistant')
