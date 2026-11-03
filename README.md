# Dense neural networks in NumPy

An educational implementation of fully connected neural networks,
backpropagation, and gradient descent using NumPy. The code accompanies
[Neural Networks: Overview](Neural%20Networks%3A%20Overview.pdf), a set of
mathematical notes on neural networks.

## Minimal example

```bash
git clone https://github.com/Hydrodynamical/neuralnetwork.git
cd neuralnetwork
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy
```

From the repository root:

```python
import numpy as np
from NetworkArch import DenseNetwork

network = DenseNetwork(arch=[10, 5, 1])
x = np.random.randn(10)
prediction = network.evaluate(x)
print(prediction)
```

## Interface

`DenseNetwork` takes a list of layer widths. Its default activation is a
scaled arctangent; custom activation and derivative functions can also be
passed to the constructor.

| Method | Purpose |
| --- | --- |
| `evaluate(input_data)` | Evaluate the network and retain the intermediate activations |
| `gradient(input_data, expected_output)` | Compute gradients for the squared-error loss |
| `do_GD(step_size)` | Update weights and biases using the stored gradients |
| `print_state()` | Print internal state for inspection |

Weights, biases, activations, pre-activation values, and their gradients are
stored on the network object. See [`NetworkArch.py`](NetworkArch.py) for the
implementation.

This is a small learning project. The implemented backend is NumPy; the
commented CuPy import is an exploratory note, not GPU support.

[License](LICENSE)
