"""
    Copyright 2013 Lyst Ltd.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""


from scipy import sparse
import numpy as np

def inner(A, B):
    if sparse.issparse(A) and sparse.issparse(B):
        return (A*B.T)[0,0]
    if not sparse.issparse(A) and not sparse.issparse(B):
        return np.inner(A, B)
    else:
        raise ValueError('sparsity of arguments is not consistant')
