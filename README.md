# Towards Quantum Machine learning with Tensor Networks

**Topics of Quantum Computing (IN2107, IN0014, IN218307)**

This project implements part of the numerical experiments proposed by Huggins et al. in the paper [Towards Quantum Machine Learning with Tensor Networks](https://arxiv.org/pdf/1803.11537.pdf).

<img src="/media/architecture.png"  width="60%">


The quantum circuits are implemented using [Qiskit](https://qiskit.org/). </br>
The tensor networks are implemented using [TensorNetwork](https://github.com/google/TensorNetwork)

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
