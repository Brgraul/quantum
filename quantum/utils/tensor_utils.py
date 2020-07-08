"""This module provides utility to generate and contract a tensor network."""

import tensornetwork as tn
from .quantum_utils import unitaries_from_weights

import numpy as np


def trigonometric_embedding(data: np.array, v: int) -> list[tn.Node]:
    """ Takes the data vector of shape 1xN and maps it to tensor_data, a vector of shape
     2x(2*N) that serves as input for the tensor network.

    The mapping used is:

    y_k = y_( k + 2*v ) = [cos( pi*x_k / 2 ), sin( pi*x_k / 2 )]

    Args:
        data: Vector of floats contained in the range [0,1]
        v: Bond dimension of the subtrees connecting the tensor network

    Returns:
        tensor_data: list of nodes holding the mapped data

    """
    feature_map = np.zeros((data.shape[0], 2))
    tensor_data = []
    for i in range(len(data) // (2 * v)):
        for k in range(2 * v):
            l = i * 2 * v + k  # Index in the input data array of length dim**2x1
            # Index in the array of dual and non-dual vectors of length 2*dim**2x1
            feature_map[l][0] = np.cos(np.pi / 2 * data[l])
            feature_map[l][1] = np.sin(np.pi / 2 * data[l])
            tensor_data.append(tn.Node(feature_map[l], name=f"data_{l}"))
        for k in range(2 * v):
            l = i * 2 * v + k
            tensor_data.append(tn.Node(feature_map[l], name=f"data_h_{l}"))
    return tensor_data


def labeling(rho: np.array):
    """ Takes a reduced density matrix rho, and returns label 1 if the first diagonal
    entry (probability of state |0>) is bigger than 0.5, and label 0 otherwise.

    Args:
        rho: Reduced density matrix of shape 2x2

    Returns:
        label: Binary label

    """
    if rho[0][0].real > 0.5:
        return 1
    else:
        return 0


def build_tensor(
    tensor_data: list[tn.Node], unitaries: list[np.Array]
) -> list[tn.Node]:
    """ Takes a valid tensor_data vector, a valid list of unitary matrices, and makes
    the necessary edge connections to construct the network as illustrated in the
    Huggins et al. paper 'Towards Quantum Machine Learning with Tensor Networks'

    arXiv:1803.11537

    Args:
        tensor_data: List of nodes holding the mapped input data in a valid format
        - see doc trigonometric_embedding()
        unitaries: List of valid unitary matrices

    Returns:
        tensor_network: List of connected tensors forming the network as explained in
        the Huggins' paper

    """

    data_length = len(tensor_data) // 2
    v = int(np.sqrt(unitaries[0].shape[0]) / 2)

    gates_per_step = []
    gates_per_step.append(data_length // (2 ** v))
    steps = int(np.log2(gates_per_step[0])) + 1  # +1 as we added the first step already
    for i in range(1, steps):
        gates_per_step.append(gates_per_step[i - 1] // 2)

    # Cumulative gates in a given step
    c_gates = [0]
    sum_ = 0
    for i in range(len(gates_per_step)):
        sum_ += 2 * gates_per_step[i]
        c_gates.append(sum_)

    tensor_network = []

    redistribute_indexes = [2 for i in range(int(np.log2(unitaries[0].size)))]

    for i, unitary in enumerate(unitaries):
        tensor_network.append(
            tn.Node(unitary.reshape(redistribute_indexes), name=f"unitary_{i}")
        )
        tensor_network.append(
            tn.Node(
                unitary.conjugate().reshape(redistribute_indexes), name=f"unitary_h_{i}"
            )
        )

    for i in range(gates_per_step[0]):
        for k in range(2 * v):
            idx = (
                2 * i * 2 * v + k
            )  # Index in the array of dual and non-dual vectors of length 2*dim**2x1
            tensor_network[2 * i][k] ^ tensor_data[idx][0]
        for k in range(2 * v):
            idx = 2 * i * 2 * v + k
            tensor_network[2 * i + 1][k] ^ tensor_data[idx + 2 * v][0]

    for i in range(
        steps - 1
    ):  #  Last execution is peeled off, as there's no 'next tensor' to link
        for j in range(gates_per_step[i]):
            # Within a given gate
            gate_idx = 2 * j + c_gates[i]
            for k in range(v):
                tensor_network[gate_idx][2 * v + k] ^ tensor_network[gate_idx + 1][
                    2 * v + k
                ]
            for k in range(v):
                tensor_network[gate_idx][k + 3 * v] ^ tensor_network[
                    ((gate_idx - c_gates[i]) // 4) * 2 + c_gates[i + 1]
                ][((gate_idx // 2) % 2) * v + k]
                tensor_network[gate_idx + 1][k + 3 * v] ^ tensor_network[
                    ((gate_idx - c_gates[i]) // 4) * 2 + c_gates[i + 1] + 1
                ][((gate_idx // 2) % 2) * v + k]
    # Contracting the last tensor in the network
    gate_idx = c_gates[-2]  #  Remember we appended an offset 0, thus -2 instead of -1
    for k in range((2 * v) - 1):
        tensor_network[gate_idx][2 * v + k] ^ tensor_network[gate_idx + 1][2 * v + k]
    return tensor_network


def contract(tensor_data, tensor_network):
    """ Takes the lists tensor_data and tensor_network, and contracts them following
    a naive algorithm.

    NOTE: This is a fragile implementation - Only works with v = 2 -pending the solution
    of the issue with Google/TensorNetwork method
    tensornetwork.contractors.auto(nodes, …)

    -> The contraction algorithm finding falls in an infinite recursion, that
    bloats the RAM and kills the Python process.


    Args:
        tensor_data: List of nodes holding the mapped input data in a valid format
        tensor_network: List of connected tensors forming the network as explained in
        the Huggins' paper

    Returns:
        res_tot.tensor: 2x2 reduced density matrix representing the quantum state at the
        end of the circuit
    """
    temp = []
    v = len(tensor_network[0].tensor.shape) // 4
    data_length = len(tensor_data) // 2

    for i in range(data_length // (2 * v)):
        cont1 = [tensor_data[i] for i in range(i * 4 * v, (i + 1) * 4 * v)]
        cont_1u = [tensor_network[i] for i in range(i * v, (i + 1) * v)]
        cont = cont1 + cont_1u
        temp.append(tn.contractors.greedy(cont, ignore_edge_order=True))

    tmp1 = temp[0] @ tensor_network[8]
    tmp2 = temp[1] @ tensor_network[9]
    res = tmp1 @ tmp2

    tmp3 = temp[2] @ tensor_network[10]
    tmp4 = temp[3] @ tensor_network[11]
    res2 = tmp3 @ tmp4

    tmp5 = res @ tensor_network[12]
    tmp6 = res2 @ tensor_network[13]
    res_tot = tmp5 @ tmp6

    return res_tot.tensor


def evaluate_tensor(image: np.array, weights: np.array, v: int) -> int:
    """ Generates a tensor network of bond dimension v and contracts it, resulting in a
        2x2 reduced density matrix
    Args:
        image: The flattened image data that serves as input to the tensor network
        weights: Parameters for creating the tensors in the network
        v: Bond dimension of the subtrees that comprise the network

    Returns:
        rho: 2x2 reduced density matrix representing the quantum state at the end of the
        circuit
    """
    unitaries = unitaries_from_weights(weights, 2 ** (2 * v))
    tensor_data = trigonometric_embedding(image, v)
    tensor_network = build_tensor(tensor_data, unitaries)
    rho = contract(tensor_data, tensor_network)
    return labeling(rho)
