import qutip
import qiskit
import numpy as np
from scipy.linalg import expm

DIMENSION = 2


def transform(x):
    feature_map = np.zeros((x.shape[0], 2))
    for index, value in enumerate(x):
        feature_map[index][0] = np.cos(np.pi / 2 * value)
        feature_map[index][1] = np.sin(np.pi / 2 * value)
    return feature_map


def tensor_dot(features):
    dot = 1
    for entry in features:
        dot = np.tensordot(dot, entry, axes=0)
    return dot


def unitary(hermitian):
    return np.matrix(expm(1j * hermitian))


def hermitian(wheights, dimension):
    diagonals = wheights[:dimension]
    dim = ((dimension**2 - dimension) // 2) + dimension
    reals = wheights[dimension:dim]
    imaginaries = wheights[dim:]
    assert reals.shape == imaginaries.shape
    H = np.diag(diagonals + 0j)
    upper = np.array([complex(a, b) for a, b in zip(reals, imaginaries)])
    lower = np.conj(upper)
    ui = np.triu_indices(dimension, 1)
    li = np.tril_indices(dimension, -1)
    H[ui] = upper
    H[li] = lower
    return H

inputs = np.random.random(DIMENSION)
wheights = np.random.random(DIMENSION**2)
u = unitary(hermitian(wheights, dimension=DIMENSION))
print(u)
feature_map = transform(inputs)
base = qiskit.QuantumCircuit(DIMENSION)
for index, state in enumerate(feature_map):
    base.initialize(state, index)
#base.unitary(u,(0,1), 'U1')
print(base)
