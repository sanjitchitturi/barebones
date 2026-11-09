"""Feedforward neural network implemented with the Python standard library.

The forward pass, squared-error loss, and backpropagation are written out as
plain loops so each step can be read on its own. Updates are online: every
training example immediately nudges the weights, in dataset order.
"""

import math
import random
from typing import List, Optional, Sequence, Tuple

Matrix = List[List[float]]
Vector = List[float]
Example = Tuple[Sequence[float], Sequence[float]]


def _sigmoid(x: float) -> float:
    """Return the logistic sigmoid of ``x``.

    Positive and negative inputs are handled separately so ``exp`` is only
    ever called on a non-positive number. That stays finite for large ``|x|``,
    where ``exp(x)`` itself would overflow.
    """
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _sigmoid_derivative(activated: float) -> float:
    """Return the sigmoid derivative from an activation, not a pre-activation.

    If ``a = sigmoid(z)`` then ``sigmoid'(z) = a * (1 - a)``.
    """
    return activated * (1.0 - activated)


def _finite_vector(values: Sequence[float], expected_len: int, label: str) -> Vector:
    """Copy ``values`` into a list of finite floats with the required length."""
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of numbers, got {values!r}.")
    try:
        length = len(values)
    except TypeError as exc:
        raise ValueError(f"{label} must be a sequence of numbers.") from exc
    if length != expected_len:
        raise ValueError(f"Expected {label} of length {expected_len}, got {length}.")

    checked: Vector = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be real numbers, got {value!r}.") from exc
        if not math.isfinite(number):
            raise ValueError(f"{label} must be finite, got {value!r}.")
        checked.append(number)
    return checked


def _positive_int(value: int, label: str) -> int:
    """Return ``value`` when it is an int greater than zero."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer, got {value!r}.")
    return value


class NeuralNetwork:
    """Fully connected sigmoid network trained by online gradient descent.

    ``weights[layer][neuron]`` lists the weights into that neuron, one per
    activation of the previous layer. ``biases[layer][neuron]`` is that
    neuron's bias. Both sequences start at the first hidden layer (or the
    output layer, when there is no hidden layer) and end at the output.

    The logged training loss is the mean, over examples, of the sum of squared
    output errors. Parameter updates follow the gradient of half of that sum,
    so the extra factor of two from differentiating a square is left out and
    absorbed into the learning rate.
    """

    def __init__(self, layer_sizes: Sequence[int], seed: Optional[int] = None) -> None:
        """Create a network and draw its initial weights and biases.

        Each value is drawn uniformly from ``[-1/sqrt(fan_in), 1/sqrt(fan_in)]``,
        where ``fan_in`` is the width of the previous layer. ``seed`` fixes the
        local random stream only; the global ``random`` module is left alone.
        """
        try:
            sizes = list(layer_sizes)
        except TypeError as exc:
            raise ValueError("layer_sizes must be a sequence of layer widths.") from exc
        if len(sizes) < 2:
            raise ValueError("Need at least an input layer and an output layer.")
        for size in sizes:
            if isinstance(size, bool) or not isinstance(size, int):
                raise ValueError(f"Layer sizes must be integers, got {size!r}.")
            if size < 1:
                raise ValueError("Every layer must have at least one neuron.")

        self.layer_sizes: List[int] = sizes
        self._rng = random.Random(seed)
        self.loss_history: List[float] = []
        self.weights: List[Matrix] = []
        self.biases: List[Vector] = []

        for fan_in, fan_out in zip(sizes, sizes[1:]):
            limit = 1.0 / math.sqrt(fan_in)
            self.weights.append(
                [
                    [self._rng.uniform(-limit, limit) for _ in range(fan_in)]
                    for _ in range(fan_out)
                ]
            )
            self.biases.append(
                [self._rng.uniform(-limit, limit) for _ in range(fan_out)]
            )

    def _forward(self, inputs: Vector) -> List[Vector]:
        """Return the activation of every layer, including the input copy."""
        if len(inputs) != self.layer_sizes[0]:
            raise ValueError(
                f"Expected {self.layer_sizes[0]} inputs, got {len(inputs)}."
            )

        activations: List[Vector] = [list(inputs)]
        for layer_w, layer_b in zip(self.weights, self.biases):
            previous = activations[-1]
            layer_out: Vector = []
            for neuron_w, bias in zip(layer_w, layer_b):
                total = bias + sum(
                    weight * activation
                    for weight, activation in zip(neuron_w, previous)
                )
                layer_out.append(_sigmoid(total))
            activations.append(layer_out)
        return activations

    def _gradients(
        self, activations: List[Vector], expected: Vector
    ) -> Tuple[List[Matrix], List[Vector]]:
        """Return weight and bias gradients for one example.

        Gradients are of ``0.5 * sum((output - target) ** 2)``. That matches
        an update of ``error * sigmoid'(activation)`` and drops the constant
        factor of two. Shapes match ``self.weights`` and ``self.biases``.
        """
        if len(expected) != self.layer_sizes[-1]:
            raise ValueError(
                f"Expected {self.layer_sizes[-1]} target values, got {len(expected)}."
            )

        num_layers = len(self.weights)
        output = activations[-1]
        deltas: List[Vector] = [[] for _ in self.weights]
        deltas[-1] = [
            (output[i] - expected[i]) * _sigmoid_derivative(output[i])
            for i in range(len(output))
        ]

        for layer in range(num_layers - 2, -1, -1):
            next_weights = self.weights[layer + 1]
            next_deltas = deltas[layer + 1]
            current = activations[layer + 1]
            layer_deltas: Vector = []
            for neuron in range(len(self.weights[layer])):
                error = sum(
                    next_weights[next_neuron][neuron] * next_deltas[next_neuron]
                    for next_neuron in range(len(next_weights))
                )
                layer_deltas.append(error * _sigmoid_derivative(current[neuron]))
            deltas[layer] = layer_deltas

        weight_grads: List[Matrix] = []
        bias_grads: List[Vector] = []
        for layer, layer_deltas in enumerate(deltas):
            incoming = activations[layer]
            weight_grads.append(
                [
                    [delta * incoming[source] for source in range(len(incoming))]
                    for delta in layer_deltas
                ]
            )
            bias_grads.append(list(layer_deltas))
        return weight_grads, bias_grads

    def _checked_examples(self, data: Sequence[Example]) -> List[Tuple[Vector, Vector]]:
        """Validate every pair before any weight is changed."""
        try:
            samples = list(data)
        except TypeError as exc:
            raise ValueError("Training data must be a sequence of examples.") from exc
        if not samples:
            raise ValueError("Training data must not be empty.")

        n_inputs = self.layer_sizes[0]
        n_outputs = self.layer_sizes[-1]
        checked: List[Tuple[Vector, Vector]] = []
        for index, sample in enumerate(samples):
            try:
                inputs, expected = sample
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Sample {index} must be an (inputs, expected) pair."
                ) from exc
            try:
                checked.append(
                    (
                        _finite_vector(inputs, n_inputs, "inputs"),
                        _finite_vector(expected, n_outputs, "target values"),
                    )
                )
            except ValueError as exc:
                raise ValueError(f"Sample {index}: {exc}") from exc
        return checked

    def predict(self, inputs: Sequence[float]) -> Vector:
        """Return the output-layer activation for one input vector."""
        checked = _finite_vector(inputs, self.layer_sizes[0], "inputs")
        return self._forward(checked)[-1]

    def loss(self, data: Sequence[Example]) -> float:
        """Return the mean sum of squared errors of the current weights.

        ``loss_history`` records a different average: the errors of the
        forward passes inside an epoch, taken before each example's update.
        """
        checked = self._checked_examples(data)
        total = 0.0
        for inputs, expected in checked:
            predicted = self._forward(inputs)[-1]
            total += sum(
                (output - target) * (output - target)
                for output, target in zip(predicted, expected)
            )
        return total / len(checked)

    def train(
        self,
        data: Sequence[Example],
        epochs: int = 5000,
        learning_rate: float = 0.5,
        print_every: int = 500,
        verbose: bool = True,
    ) -> List[float]:
        """Train on ``data`` and return the per-epoch training loss.

        Each example is forwarded, its gradient is applied, and only then does
        the next example run. The loss appended for an epoch is the average of
        those pre-update squared errors, not a separate pass after the epoch.

        ``print_every`` must be a positive integer when ``verbose`` is true.
        Set ``verbose`` to false to keep training silent. ``learning_rate``
        must be finite and greater than or equal to zero.
        """
        epochs = _positive_int(epochs, "epochs")
        if isinstance(learning_rate, bool) or not isinstance(
            learning_rate, (int, float)
        ):
            raise ValueError(
                f"learning_rate must be a real number, got {learning_rate!r}."
            )
        if not math.isfinite(learning_rate) or learning_rate < 0.0:
            raise ValueError(
                f"learning_rate must be finite and >= 0, got {learning_rate!r}."
            )
        if verbose and (
            isinstance(print_every, bool)
            or not isinstance(print_every, int)
            or print_every < 1
        ):
            raise ValueError(
                "print_every must be a positive integer when verbose is true, "
                f"got {print_every!r}."
            )

        checked = self._checked_examples(data)
        self.loss_history = []

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for inputs, expected in checked:
                activations = self._forward(inputs)
                output = activations[-1]
                total_loss += sum(
                    (prediction - target) * (prediction - target)
                    for prediction, target in zip(output, expected)
                )
                weight_grads, bias_grads = self._gradients(activations, expected)
                next_weights: List[Matrix] = []
                next_biases: List[Vector] = []
                for layer in range(len(self.weights)):
                    layer_weights: Matrix = []
                    layer_biases: Vector = []
                    for neuron in range(len(self.weights[layer])):
                        neuron_weights: Vector = []
                        for source in range(len(self.weights[layer][neuron])):
                            step = learning_rate * weight_grads[layer][neuron][source]
                            candidate = self.weights[layer][neuron][source] - step
                            if not math.isfinite(candidate):
                                raise FloatingPointError("Nonfinite weight update; reduce the learning rate or input scale.")
                            neuron_weights.append(candidate)
                        bias = self.biases[layer][neuron] - learning_rate * bias_grads[layer][neuron]
                        if not math.isfinite(bias):
                            raise FloatingPointError("Nonfinite bias update; reduce the learning rate or input scale.")
                        layer_weights.append(neuron_weights)
                        layer_biases.append(bias)
                    next_weights.append(layer_weights)
                    next_biases.append(layer_biases)
                self.weights, self.biases = next_weights, next_biases

            average_loss = total_loss / len(checked)
            self.loss_history.append(average_loss)
            if verbose and (
                epoch == 1 or epoch == epochs or epoch % print_every == 0
            ):
                print(f"  Epoch {epoch:>5d}  |  Loss: {average_loss:.6f}")

        return self.loss_history
