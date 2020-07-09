# Towards Quantum Machine learning with Tensor Networks

**Topics of Quantum Computing (IN2107, IN0014, IN218307)**

This project implements part of the numerical experiments proposed by Huggins et al. in the paper [Towards Quantum Machine Learning with Tensor Networks](https://arxiv.org/pdf/1803.11537.pdf). The experiments implemented so far have been:

(a) Classification task with a quantum circuit doing the predictions 
(b) Classification task with an equivalent tensor network doing the predictions.

<img src="/media/labeled_circuit.png"  width="60%" style="padding-bottom: 5%;padding-top: 2%;">


The quantum circuits are implemented using [Qiskit](https://qiskit.org/).
The tensor networks are implemented using [TensorNetwork](https://github.com/google/TensorNetwork)

## Run w/o Install

You can get a taste of the tensor algorithms used in this project with zero setup pain by accessing the following Google Colab notebooks:

- [Functional TN notebook](https://colab.research.google.com/drive/1YAwQNKy3afOqCqWn31j_r3qZ19jWhmLA?usp=sharing)

- [Monolith TN notebook](https://colab.research.google.com/drive/122IO8YMrWMxiPwIshLCMNG3wjbQO_rvG?usp=sharing)

## Installation

Install the dependencies from the requirements file by using [pip](https://pip.pypa.io/en/stable/)

```bash
pip install -r requirements.txt
```

Unless wanting to make a global install, we encourage to first create a virtual environment and install only after activating it.

## Usage

The project contains a set of notebooks that allow you to play with the implementation of the tensor network plus binary classification example based on the MNIST dataset, and the necessary quantum circuit, tensor network and data utility functions.

### Notebooks

For executing the notebooks, simply enter the ```/notebooks``` folder and launch jupyter lab.

```bash
cd notbooks
jupyter lab
```

### Classification architecture

If you wish to run the classification task with the tensor network prediction, pass the argument '-t' to the module.

```
python -m quantum.network -t
```

<img src="/media/architecture_tensor.png"  width="60%" style="padding-bottom: 3%;padding-top: 2%;">

To run the example with quantum circuit predictions, pass no arguments to the module.

```
python -m quantum.network 
```

<img src="/media/architecture_circuit.png"  width="60%" style="padding-top: 2%;">

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

## Roadmap
The project is still under development. We aim to generalize the examples presented, and incorporate more lightweight networks like the MPS base one presented by Stavros et al. in [TensorNetwork for Machine Learning](https://arxiv.org/abs/1906.06329). 
Next milestones are:
- Parametrize the naive tensor network contraction for v!=2
- Implement more resource efficient network contractions.
- Implement lighter tensor network architectures (MPS).
- Adapt the implementation for the non-binary classification task.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. 

## License
[MIT](https://choosealicense.com/licenses/mit/)