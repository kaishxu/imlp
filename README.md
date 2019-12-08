# interval Multilayer Perceptron

Interval Multilayer Perceptron is a method which implements multilayer perceptron with interval-valued inputs and interval-valued outputs. This package is based on the paper, [**iMLP**: Applying multi-layer perceptrons to interval-valued data](https://link.springer.com/article/10.1007/s11063-007-9035-z), [AM San Roque](https://scholar.google.com/citations?user=Kfu7yNMAAAAJ&hl=en&oi=sra).



## Usage

```python
from imlp import iAct, iLoss, get_model
import numpy as np

# Generate the synthetic data
x1 = np.sin(np.arange(0, 9, 0.01))
x2 = np.cos(np.arange(0, 9, 0.01))
x3 = x1**2
x4 = (x1+x2)/2
tmp = np.ones((900,))

Xtrain_c = x3[0:5]
Xtrain_r = tmp[0:5]
Ytrain_c = x4[0:1]
Ytrain_r = tmp[0:1]

for i in range(1,100):
    Xtrain_c = np.vstack((Xtrain_c, x3[i:i+5]))
    Xtrain_r = np.vstack((Xtrain_r, tmp[i:i+5]))
    Ytrain_c = np.vstack((Ytrain_c, x4[i:i+1]))
    Ytrain_r = np.vstack((Ytrain_r, tmp[i:i+1]))

# Parameters
input_dim = 5
output_dim = 1
num_hidden_layers = 2
num_units = [200, 200]  # only for hidden layers (not output layer)
act = ['relu', 'relu']  # only for hidden layers (not output layer)
beta = 0.5  # control the balance between center and radius in loss function

# Get model
model = get_model(input_dim, output_dim, num_units, act, beta, num_hidden_layers)

# Train
model.fit(x=[Xtrain_c, Xtrain_r], y=[Ytrain_c, Ytrain_r], epochs=10)
```

## Structure of the iMLP

<img src="https://github.com/KaishuaiXu/imlp/blob/master/pic/structure.png?raw=true" alt="kernel function" width="577" height="245" />



## Hidden layer

<img src="https://github.com/KaishuaiXu/imlp/blob/master/pic/hidden%20layer.png?raw=true" alt="kernel function" width="425" height="73.5" />



## Activation function

<img src="https://github.com/KaishuaiXu/imlp/blob/master/pic/activation.png?raw=true" alt="kernel function" width="361.5" height="156" />



## Loss function

<img src="https://github.com/KaishuaiXu/imlp/blob/master/pic/loss%20function.png?raw=true" alt="kernel function" width="411" height="65" />

