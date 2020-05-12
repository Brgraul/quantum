import numpy as np
from scipy.linalg import expm

import qiskit
from qiskit import Aer
from qiskit.quantum_info.operators.predicates import (is_hermitian_matrix,
                                                      is_unitary_matrix)


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


def unitaries_from_weights(weights):
    unitaries = []
    for weight in weights:
        unitaries.append(
            unitary_from_hermitian(hermitian_from_weights(weight,
                                                          dimension=4)))
    return unitaries


def run_circuit(image,
                weights,
                backend=Aer.get_backend('qasm_simulator'),
                draw=False):
    dimension = len(image)
    features = transform(image)
    unitaries = unitaries_from_weights(weights)

    classic_circuit = qiskit.ClassicalRegister(1)
    quantum_circuit = qiskit.QuantumRegister(dimension)
    base_circiut = qiskit.QuantumCircuit(quantum_circuit, classic_circuit)

    for index, state in enumerate(features):
        base_circiut.initialize(state, index)

    index = 0
    for i in range(int(np.log2(dimension))):
        steps = dimension // (2**(i + 1))
        step_size = dimension // steps

        for j in range(steps):
            qubits = []
            lower = step_size * j + 2**i - 1
            upper = step_size * j + step_size
            qubits.append(base_circiut.qubits[lower])
            qubits.append(base_circiut.qubits[upper - 1])
            base_circiut.unitary(unitaries[index], qubits, f'U({i},{j})')
            index += 1
    base_circiut.measure([dimension - 1], [0])

    if draw:
        print(base_circiut)
    result = qiskit.execute(base_circiut, backend).result()
    return result
