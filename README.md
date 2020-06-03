# Quantum Machine Learning Example

**Under developement**

This project implements a disciminative algorithm for quantum machine learning
described in [Towards Quantum Machine Learning with Tensor Networks](https://arxiv.org/pdf/1803.11537.pdf).
The quantum circuits are implemented using [Qiskit](https://qiskit.org/).

## Install

Simply clone the repository and install the necessary requirements
```
    git clone https://github.com/Gistbatch/quantum.git
    cd quantum
    python -m venv venv
    venv\Scripts\activate (Windows)
    venv\bin\activate (Linux)
    python -m pip install --upgrade pip
```

The project dependencies are changing regularly at the moment.
```
    pip install tensorflow numpy matplotlib qiskit
```

## Run
As an example you can run the network file as an module
```
    python -m quantum.network
```

### Availabel Ciruits
1. Default Circuit using unitaries for two Qubits at a time described in figure 7.

    ![Default circuit](https://github.com/Gistbatch/quantum/blob/master/quantum/data/default.png)
2. Efficient Circuit using unitaries for four Quibts at a time descibed in figure 11.

    ![Efficient circiut](https://github.com/Gistbatch/quantum/blob/master/quantum/data/efficient.png)

## Sources
---