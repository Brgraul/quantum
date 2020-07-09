# Towards Quantum Machine learning with Tensor Networks

**Topics of Quantum Computing (IN2107, IN0014, IN218307)**

This project implements part of the numerical experiments proposed by Huggins et al. in the paper [Towards Quantum Machine Learning with Tensor Networks](https://arxiv.org/pdf/1803.11537.pdf). The experiments implemented so far have been:

(a) Classification task with a quantum circuit doing the predictions 
(b) Classification task with an equivalent tensor network doing the predictions.

<img src="/media/labeled_circuit.png"  width="60%" style="padding-bottom: 5%;padding-top: 2%;">


The quantum circuits are implemented using [Qiskit](https://qiskit.org/).
The tensor networks are implemented using [TensorNetwork](https://github.com/google/TensorNetwork)

## Installation

Install the dependencies from the requirements file by using [pip](https://pip.pypa.io/en/stable/)

```bash
pip install -r requirements.txt
```

Unless wanting to make a global install, we encourage to first create a virtual environment and install only after activating it.

## Usage

The project contains a set of notebooks that allow you to play with the implementation of the tensor network plus a full classification architecture, and the necessary quantum circuit, tensor network and data utility functions.

### Notebooks

For executing the notebooks, simply enter the ```/notebooks``` folder and launch jupyter lab.

```bash
cd notbooks
jupyter lab
```

### Classification architecture

```
python -m quantum.network
```

## Project structure 

- **/notebooks**: Holds the experiments used for the development of the tensor network code and an isolated example of its functioning. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **AltContraction.ipynb**: Monolith implementation of the tensor network contraction and labeling. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -  **TensorNetwork.ipynb**: Monolith relying on the method tensornetwork.contractors.auto() </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -  **RefactorTest.ipynb**: Functional refactoring of the tensor network contraction and labeling. </br>
- **/quantum**: Holds the codebase for the classification algorithms: </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **network.py**: Holds the QuantumNetwork class with its respective training and prediction methods. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **/utils**: Holds the scripts supporting the functioning of the network. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  - **quantum_utils.py**: Functions for unitary matrix and quantum circuit generation. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  - **tensor_utils.py**: Functions for the embedding, generation of the network, contraction and labeling </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  - **data_utils.py**: Preprocessing functions for loading the classic data into the network. </br>



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
