# Quantum Machine Learning Example

**Under developement**

This project implements a disciminative algorithm for quantum machine learning
described in [Towards Quantum Machine Learning with Tensor Networks](https://arxiv.org/pdf/1803.11537.pdf).
The quantum circuits are implemented using [Qiskit](https://qiskit.org/), this
imay change.

## Install

Simply clone the repository and install the necessary requirements
```
    git clone https://github.com/Gistbatch/quantum.git
    cd quantum
    python -m venv venv
    venv\Scripts\activate
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

## Sources
---