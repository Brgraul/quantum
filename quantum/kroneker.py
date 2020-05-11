import numpy as np

Y = np.matrix([[0, -1j],[1j, 0]])
Z = np.matrix([[1,0],[0,-1]])
result = np.kron(Y,Z)


print(f'{Y}\n⊗\n{Z}\n=\n{result}')