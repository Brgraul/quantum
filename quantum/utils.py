import numpy as np
from scipy.linalg import expm

import qiskit
from qiskit.quantum_info.operators.predicates import (is_hermitian_matrix,
                                                      is_unitary_matrix)

DIMENSION = 8


def transform(x):
    feature_map = np.zeros((x.shape[0], 2))
    for index, value in enumerate(x):
        feature_map[index][0] = np.cos(np.pi / 2 * value)
        feature_map[index][1] = np.sin(np.pi / 2 * value)
    return feature_map


def unitary_from_hermitian(hermitian):
    U = np.matrix(expm(1j * hermitian))
    assert is_unitary_matrix(U)
    return U


def hermitian_from_weights(weights, dimension):
    diagonals = weights[:dimension]
    dim = ((dimension**2 - dimension) // 2) + dimension
    reals = weights[dimension:dim]
    imaginaries = weights[dim:]
    assert reals.shape == imaginaries.shape
    H = np.matrix(np.diag(diagonals + 0j))
    H[np.triu_indices(dimension, 1)] = np.array(
        [complex(a, b) for a, b in zip(reals, imaginaries)])
    H = H + H.H  # tril and triu don't use the same ordering!
    assert is_hermitian_matrix(H)
    return H


if __name__ == "__main__":
    inputs = np.random.random(DIMENSION)
    feature_map = transform(inputs)
    weights = []
    unitaries = []
    for i in range(DIMENSION - 1):
        weight = np.random.random(4**2)
        weights.append(weight)
        unitaries.append(
            unitary_from_hermitian(hermitian_from_weights(weight,
                                                          dimension=4)))

    base = qiskit.QuantumCircuit(DIMENSION, 1)
    for index, state in enumerate(feature_map):
        base.initialize(state, index)

    index = 0
    for i in range(int(np.log2(DIMENSION))):
        steps = DIMENSION // (2**(i + 1))
        step_size = DIMENSION // steps

        for j in range(steps):
            qubits = []
            lower = step_size * j + 2**i - 1
            upper = step_size * j + step_size
            qubits.append(base.qubits[lower])
            qubits.append(base.qubits[upper - 1])
            base.unitary(unitaries[index], qubits, f'U({i},{j})')
            index += 1
    base.measure([DIMENSION - 1], [0])
    print(base)
