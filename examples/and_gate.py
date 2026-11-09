"""Train one sigmoid layer on AND, which is linearly separable.

Run this from the repository root:

    python examples/and_gate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from network import NeuralNetwork

AND = [
    ([0, 0], [0]),
    ([0, 1], [0]),
    ([1, 0], [0]),
    ([1, 1], [1]),
]


def main() -> None:
    print("AND gate, no hidden layer")
    print("Architecture: 2 inputs, 1 output")
    print()

    network = NeuralNetwork([2, 1], seed=0)
    network.train(AND, epochs=2000, learning_rate=1.0, print_every=1000)

    print()
    print("Predictions after training:")
    print(f"  {'Input':<12}{'Expected':<12}{'Predicted':<12}{'Rounded'}")
    for inputs, expected in AND:
        predicted = network.predict(inputs)[0]
        print(
            f"  {str(inputs):<12}"
            f"{expected[0]:<12}"
            f"{predicted:<12.4f}"
            f"{round(predicted)}"
        )


if __name__ == "__main__":
    main()
