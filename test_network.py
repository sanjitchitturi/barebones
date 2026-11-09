"""Unit tests for the from-scratch network, including a gradient check."""

import contextlib
import io
import math
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Callable, List

from network import NeuralNetwork, _sigmoid, _sigmoid_derivative

XOR = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0]),
]


def _central_difference(
    values: List[float], index: int, epsilon: float, loss: Callable[[], float]
) -> float:
    """Derivative of ``loss`` with respect to ``values[index]``."""
    original = values[index]
    values[index] = original + epsilon
    loss_plus = loss()
    values[index] = original - epsilon
    loss_minus = loss()
    values[index] = original
    return (loss_plus - loss_minus) / (2.0 * epsilon)


class TestSigmoid(unittest.TestCase):
    def test_midpoint_and_symmetry(self) -> None:
        self.assertEqual(_sigmoid(0.0), 0.5)
        for value in (0.2, 1.3, 4.0, 40.0, 500.0):
            self.assertAlmostEqual(_sigmoid(value) + _sigmoid(-value), 1.0, places=12)

    def test_extremes_stay_finite(self) -> None:
        self.assertGreater(_sigmoid(1000.0), 1.0 - 1e-9)
        self.assertLess(_sigmoid(-1000.0), 1e-9)
        self.assertTrue(math.isfinite(_sigmoid(1000.0)))
        self.assertTrue(math.isfinite(_sigmoid(-1000.0)))

    def test_derivative_matches_finite_difference(self) -> None:
        epsilon = 1e-6
        for value in (-2.0, -0.5, 0.0, 0.3, 1.5, 3.0):
            activated = _sigmoid(value)
            numeric = (_sigmoid(value + epsilon) - _sigmoid(value - epsilon)) / (2.0 * epsilon)
            self.assertAlmostEqual(_sigmoid_derivative(activated), numeric, places=6)


class TestNeuralNetworkInit(unittest.TestCase):
    def test_creates_correct_layer_sizes(self) -> None:
        network = NeuralNetwork([2, 3, 1], seed=0)
        self.assertEqual(network.layer_sizes, [2, 3, 1])
        self.assertEqual(len(network.weights), 2)
        self.assertEqual(len(network.biases), 2)
        self.assertEqual(len(network.weights[0]), 3)
        self.assertEqual(len(network.weights[0][0]), 2)
        self.assertEqual(len(network.biases[0]), 3)
        self.assertEqual(len(network.weights[1]), 1)
        self.assertEqual(len(network.weights[1][0]), 3)
        self.assertEqual(len(network.biases[1]), 1)

    def test_layer_sizes_are_copied(self) -> None:
        sizes = [2, 3, 1]
        network = NeuralNetwork(sizes, seed=0)
        sizes.append(4)
        self.assertEqual(network.layer_sizes, [2, 3, 1])

    def test_initial_parameters_respect_fan_in_bound(self) -> None:
        sizes = [4, 3, 2]
        network = NeuralNetwork(sizes, seed=5)
        for layer, fan_in in enumerate(sizes[:-1]):
            limit = 1.0 / math.sqrt(fan_in)
            for row in network.weights[layer]:
                for weight in row:
                    self.assertGreaterEqual(weight, -limit)
                    self.assertLessEqual(weight, limit)
            for bias in network.biases[layer]:
                self.assertGreaterEqual(bias, -limit)
                self.assertLessEqual(bias, limit)

    def test_same_seed_matches_and_different_seed_differs(self) -> None:
        first = NeuralNetwork([2, 3, 1], seed=7)
        second = NeuralNetwork([2, 3, 1], seed=7)
        third = NeuralNetwork([2, 3, 1], seed=8)
        self.assertEqual(first.weights, second.weights)
        self.assertEqual(first.biases, second.biases)
        self.assertNotEqual(first.weights, third.weights)

    def test_rejects_bad_architectures(self) -> None:
        for sizes in ([3], [2, 0, 1], [2, -1, 1], [2, 1.5, 1], [2, True, 1], []):
            with self.subTest(sizes=sizes):
                with self.assertRaises(ValueError):
                    NeuralNetwork(sizes)
        with self.assertRaises(ValueError):
            NeuralNetwork(5)  # type: ignore[arg-type]


class TestPredict(unittest.TestCase):
    def test_output_length_matches_last_layer(self) -> None:
        network = NeuralNetwork([3, 5, 2], seed=0)
        result = network.predict([1.0, 2.0, 3.0])
        self.assertEqual(len(result), 2)

    def test_outputs_are_between_zero_and_one(self) -> None:
        network = NeuralNetwork([2, 4, 1], seed=0)
        result = network.predict([0.5, 0.5])
        for value in result:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_wrong_input_size_raises(self) -> None:
        network = NeuralNetwork([2, 3, 1], seed=0)
        with self.assertRaises(ValueError):
            network.predict([1.0])
        with self.assertRaises(ValueError):
            network.predict([1.0, 2.0, 3.0])

    def test_rejects_non_finite_and_non_numeric_inputs(self) -> None:
        network = NeuralNetwork([2, 3, 1], seed=0)
        for inputs in ([math.nan, 0.0], [math.inf, 0.0], ["x", 0.0], "ab"):
            with self.subTest(inputs=inputs):
                with self.assertRaises(ValueError):
                    network.predict(inputs)  # type: ignore[arg-type]

    def test_reproducibility_with_same_seed(self) -> None:
        first = NeuralNetwork([2, 3, 1], seed=7)
        second = NeuralNetwork([2, 3, 1], seed=7)
        self.assertEqual(first.predict([1, 0]), second.predict([1, 0]))

    def test_predict_does_not_require_a_list(self) -> None:
        network = NeuralNetwork([2, 1], seed=1)
        self.assertEqual(len(network.predict((0.2, 0.4))), 1)


class TestGradients(unittest.TestCase):
    def test_gradients_match_finite_differences(self) -> None:
        architectures = ([2, 1], [2, 3, 2], [3, 2, 2, 1])
        inputs = {
            2: [0.2, -0.4],
            3: [0.5, -1.0, 0.25],
        }
        expected = {
            1: [0.8],
            2: [0.1, 0.7],
        }
        epsilon = 1e-5
        for sizes in architectures:
            with self.subTest(sizes=sizes):
                network = NeuralNetwork(sizes, seed=1)
                sample_in = inputs[sizes[0]]
                sample_out = expected[sizes[-1]]
                activations = network._forward(sample_in)
                weight_grads, bias_grads = network._gradients(activations, sample_out)

                largest = max(
                    abs(value)
                    for layer in weight_grads
                    for row in layer
                    for value in row
                )
                self.assertGreater(largest, 1e-6)

                def half_sse() -> float:
                    # Updates follow the gradient of half the squared error.
                    predicted = network.predict(sample_in)
                    return 0.5 * sum(
                        (output - target) * (output - target)
                        for output, target in zip(predicted, sample_out)
                    )

                for layer in range(len(network.weights)):
                    for neuron in range(len(network.weights[layer])):
                        for source in range(len(network.weights[layer][neuron])):
                            numeric = _central_difference(
                                network.weights[layer][neuron], source, epsilon, half_sse
                            )
                            self.assertAlmostEqual(
                                weight_grads[layer][neuron][source], numeric, delta=1e-6
                            )
                        numeric = _central_difference(
                            network.biases[layer], neuron, epsilon, half_sse
                        )
                        self.assertAlmostEqual(bias_grads[layer][neuron], numeric, delta=1e-6)

    def test_one_step_applies_those_gradients(self) -> None:
        network = NeuralNetwork([2, 3, 2], seed=2)
        inputs = [0.3, 0.9]
        expected = [0.2, 0.5]
        learning_rate = 0.25
        activations = network._forward(inputs)
        weight_grads, bias_grads = network._gradients(activations, expected)
        before_weights = [
            [row[:] for row in layer] for layer in network.weights
        ]
        before_biases = [row[:] for row in network.biases]

        network.train([(inputs, expected)], epochs=1, learning_rate=learning_rate, verbose=False)

        for layer in range(len(before_weights)):
            for neuron in range(len(before_weights[layer])):
                for source in range(len(before_weights[layer][neuron])):
                    expected_weight = (
                        before_weights[layer][neuron][source]
                        - learning_rate * weight_grads[layer][neuron][source]
                    )
                    self.assertAlmostEqual(
                        network.weights[layer][neuron][source], expected_weight, places=12
                    )
                expected_bias = (
                    before_biases[layer][neuron]
                    - learning_rate * bias_grads[layer][neuron]
                )
                self.assertAlmostEqual(
                    network.biases[layer][neuron], expected_bias, places=12
                )


class TestTraining(unittest.TestCase):
    def test_overflowing_update_leaves_parameters_unchanged(self) -> None:
        network = NeuralNetwork([1, 1], seed=1)
        before_weights = [[row[:] for row in layer] for layer in network.weights]
        before_biases = [layer[:] for layer in network.biases]

        with self.assertRaises(FloatingPointError):
            network.train([([0.0], [10.0])], epochs=1, learning_rate=1e308, verbose=False)

        self.assertEqual(network.weights, before_weights)
        self.assertEqual(network.biases, before_biases)

    def test_loss_decreases(self) -> None:
        network = NeuralNetwork([2, 4, 1], seed=42)
        history = network.train(XOR, epochs=500, learning_rate=2.0, verbose=False)
        self.assertEqual(len(history), 500)
        self.assertGreater(history[0], history[-1])
        self.assertEqual(history, network.loss_history)

    def test_xor_convergence(self) -> None:
        network = NeuralNetwork([2, 4, 1], seed=42)
        network.train(XOR, epochs=3000, learning_rate=2.0, verbose=False)
        for inputs, expected in XOR:
            self.assertEqual(round(network.predict(inputs)[0]), expected[0])

    def test_and_without_hidden_layer(self) -> None:
        network = NeuralNetwork([2, 1], seed=0)
        data = [([0, 0], [0]), ([0, 1], [0]), ([1, 0], [0]), ([1, 1], [1])]
        history = network.train(data, epochs=2000, learning_rate=1.0, verbose=False)
        self.assertGreater(history[0], history[-1])
        for inputs, expected in data:
            self.assertEqual(round(network.predict(inputs)[0]), expected[0])

    def test_loss_is_mean_sum_of_squares(self) -> None:
        network = NeuralNetwork([2, 2, 2], seed=1)
        data = [([0.0, 1.0], [0.0, 1.0]), ([1.0, 0.0], [1.0, 0.0])]
        manual = 0.0
        for inputs, expected in data:
            predicted = network.predict(inputs)
            manual += sum((output - target) ** 2 for output, target in zip(predicted, expected))
        manual /= len(data)
        self.assertAlmostEqual(network.loss(data), manual, places=12)

    def test_training_is_reproducible(self) -> None:
        first = NeuralNetwork([2, 3, 1], seed=4)
        second = NeuralNetwork([2, 3, 1], seed=4)
        data = [([0.0, 0.0], [0.0]), ([1.0, 1.0], [1.0])]
        first.train(data, epochs=20, learning_rate=0.7, verbose=False)
        second.train(data, epochs=20, learning_rate=0.7, verbose=False)
        self.assertEqual(first.weights, second.weights)
        self.assertEqual(first.loss_history, second.loss_history)

    def test_empty_data_raises(self) -> None:
        network = NeuralNetwork([2, 3, 1], seed=0)
        with self.assertRaises(ValueError):
            network.train([], epochs=10, learning_rate=0.5, verbose=False)
        with self.assertRaises(ValueError):
            network.loss([])

    def test_bad_sample_shape_raises_without_changing_weights(self) -> None:
        network = NeuralNetwork([2, 2, 1], seed=0)
        network.train(XOR, epochs=1, learning_rate=0.1, verbose=False)
        saved_weights = [[row[:] for row in layer] for layer in network.weights]
        saved_history = list(network.loss_history)
        bad_sets = [
            [([0, 0], [0]), ([1], [1])],
            [([0, 0, 1], [0])],
            [([0, 0], [0, 1])],
            [([0, 0], [])],
            [([math.nan, 0], [0])],
            [([0, 0], [math.inf])],
        ]
        for data in bad_sets:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    network.train(data, epochs=5, learning_rate=0.5, verbose=False)
                self.assertEqual(network.weights, saved_weights)
                self.assertEqual(network.loss_history, saved_history)

        with self.assertRaisesRegex(ValueError, "Sample 1"):
            network.train([([0, 0], [0]), ([1], [1])], epochs=1, verbose=False)

    def test_rejects_bad_training_arguments(self) -> None:
        network = NeuralNetwork([2, 1], seed=0)
        data = [([0.0, 1.0], [1.0])]
        before = [row[:] for row in network.weights]
        cases = [
            {"epochs": 0},
            {"epochs": -3},
            {"epochs": True},
            {"learning_rate": -0.1},
            {"learning_rate": math.nan},
            {"learning_rate": math.inf},
            {"print_every": 0},
            {"print_every": -5},
        ]
        for kwargs in cases:
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    network.train(data, verbose=True, **kwargs)  # type: ignore[arg-type]
                self.assertEqual(network.weights, before)
                self.assertEqual(network.loss_history, [])

    def test_verbose_false_is_silent_and_true_prints_ends(self) -> None:
        data = [([0.0, 0.0], [0.0]), ([1.0, 1.0], [1.0])]
        quiet = NeuralNetwork([2, 2, 1], seed=0)
        with contextlib.redirect_stdout(io.StringIO()) as buffer:
            quiet.train(data, epochs=3, learning_rate=0.5, verbose=False, print_every=0)
        self.assertEqual(buffer.getvalue(), "")

        noisy = NeuralNetwork([2, 2, 1], seed=0)
        with contextlib.redirect_stdout(io.StringIO()) as buffer:
            noisy.train(data, epochs=3, learning_rate=0.5, verbose=True, print_every=100)
        text = buffer.getvalue()
        self.assertIn("Epoch     1", text)
        self.assertIn("Epoch     3", text)
        self.assertNotIn("Epoch     2", text)


class TestDemos(unittest.TestCase):
    def test_readme_sample_matches_xor_demo(self) -> None:
        root = Path(__file__).resolve().parent
        readme = (root / "README.md").read_text(encoding="utf-8")
        marker = "```\nBarebones Neural Network\n"
        start = readme.index(marker) + len("```\n")
        end = readme.index("\n```", start)
        sample = readme[start:end].strip("\n")

        from main import main

        with contextlib.redirect_stdout(io.StringIO()) as buffer:
            main()
        self.assertEqual(buffer.getvalue().strip("\n"), sample)

    def test_and_example_learns_the_gate(self) -> None:
        root = Path(__file__).resolve().parent
        completed = subprocess.run(
            [sys.executable, str(root / "examples" / "and_gate.py")],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        rounded = [
            int(line.split()[-1])
            for line in completed.stdout.splitlines()
            if line.startswith("  [")
        ]
        self.assertEqual(rounded, [0, 0, 0, 1])
        self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()
