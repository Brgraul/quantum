"""This module provides utility to run a quantum circuit."""
import numpy as np
from scipy.linalg import expm

import qiskit
from qiskit import Aer
from qiskit.quantum_info.operators.predicates import (is_hermitian_matrix,
                                                      is_unitary_matrix)


def transform(input_):
    """Transforms a 1-dimensional array into a 2-dimensianl feature array.
        The transformation is done by [cos(pi/2*x), sin(pi/2*x)]

    Args:
        input: The 1D input.

    Returns:
        featrue_map: 2D transformed array.
    """
    feature_map = np.zeros((input_.shape[0], 2))
    for index, value in enumerate(input_):
        feature_map[index][0] = np.cos(np.pi / 2 * value)
        feature_map[index][1] = np.sin(np.pi / 2 * value)
    return feature_map


def unitary_from_hermitian(hermitian):
    """Generates a unitary matrix from a hermitian matrix.
        The formula is U = e^(i*H).

    Args:
        hermitian: A hermitian matrix.

    Returns:
        unitary: The resulting unitarian matrix.

    Raises:
        AssertionError: If the resulting matrix is not unitarian.
    """
    unitary = np.matrix(expm(1j * hermitian))
    assert is_unitary_matrix(unitary)
    return unitary


def hermitian_from_weights(weights, dimension):
    """Generates a  complex hermitian matrix from a set of weights.
        The hermitian is constructed by an upper triangle matrix which then is
        added to its transpose. The first dimension weights are used for the real
        diagonal values, the next values are used for the real parts of the upper
        triangle the rest for the imaginarie parts.

    Args:
        weights: List of weights.
        dimension: size of the matrix.

    Returns:
        hermitian: The resulting hermitian matrix.

    Raises:
        AssertionError: If the resulting matrix is not hermitian.
    """
    diagonals = weights[:dimension]
    dim = ((dimension**2 - dimension) // 2) + dimension
    reals = weights[dimension:dim]
    imaginaries = weights[dim:]
    assert reals.shape == imaginaries.shape
    hermitian = np.matrix(np.diag(diagonals + 0j))
    hermitian[np.triu_indices(dimension, 1)] = np.array(
        [complex(a, b) for a, b in zip(reals, imaginaries)])
    hermitian = hermitian + hermitian.H  # tril and triu don't use the same ordering!
    assert is_hermitian_matrix(hermitian)
    return hermitian


def unitaries_from_weights(weights):
    """Wrapper function to generate unitary matricies from weight list.

    Args:
        weights: The weights to transform.

    Returns:
        unitaries: The resulting list of unitary matrices.
    """
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
    """Executes a quantum circiut on the image using the weights for unitary
        operators.

    Args:
        image: The flat input image.
        weights: List of weights for unitaries.
        backend: Simulation backend.
        draw (bool): Draw the circuit.

    Returns:
        counts: The simulated measurement counts.
    """
    dimension = len(image)
    features = transform(image)
    unitaries = unitaries_from_weights(weights)

    base_circiut = qiskit.QuantumCircuit(dimension, 1)

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
    counts = qiskit.execute(base_circiut, backend).result().get_counts()
    return counts