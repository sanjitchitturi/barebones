# Barebones

Train a small feedforward neural network using Python lists and the standard library. The XOR example uses a hidden layer. The AND example shows what a single sigmoid neuron can learn.

## Quick start

From the repository root, run:

```sh
python3 main.py
python3 examples/and_gate.py
python3 -m unittest discover -v
```

Each example prints predictions for the four truth-table rows. Both use fixed seeds. In `network.py`, the forward pass stores activations and the backward pass computes weight and bias gradients. Training updates parameters after each example, in input order.

Use `NeuralNetwork([2, 4, 1], seed=42)` to create the XOR architecture. Call `train(examples)` with pairs of input and target vectors, then `predict(inputs)` to get output activations. The training history records errors before each update. Call `loss(examples)` for the error of the final weights.
If a learning rate or input scale would create a nonfinite parameter, training
raises `FloatingPointError` before applying that example's update. Earlier
examples in the run may already have updated the network.

## Verification

Run `python3 -m unittest discover -v`. The suite checks gradients, input validation, and the recorded example output.

## Recorded XOR run

```
Barebones Neural Network
Architecture: 2 inputs, 4 hidden neurons, 1 output

Predictions before training:
  Input       Expected    Predicted   Rounded
  [0, 0]      0           0.3722      0
  [0, 1]      1           0.3964      0
  [1, 0]      1           0.3758      0
  [1, 1]      0           0.4000      0

  Epoch     1  |  Loss: 0.313241
  Epoch  1000  |  Loss: 0.001109
  Epoch  2000  |  Loss: 0.000414
  Epoch  3000  |  Loss: 0.000251
  Epoch  4000  |  Loss: 0.000179
  Epoch  5000  |  Loss: 0.000139

Predictions after training:
  Input       Expected    Predicted   Rounded
  [0, 0]      0           0.0051      0
  [0, 1]      1           0.9886      1
  [1, 0]      1           0.9891      1
  [1, 1]      0           0.0167      0

Loss fell from 0.3132 to 0.000139 over 5000 epochs.
```

## License

[MIT License](LICENSE) for the project code. Third-party software, models, and datasets retain their own terms.
