"""Train a small network on XOR and print predictions before and after."""

from network import NeuralNetwork

XOR = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0]),
]


def show_predictions(network: NeuralNetwork, title: str) -> None:
    """Print one row per XOR input."""
    print(title)
    print(f"  {'Input':<12}{'Expected':<12}{'Predicted':<12}{'Rounded'}")
    for inputs, expected in XOR:
        predicted = network.predict(inputs)[0]
        print(
            f"  {str(inputs):<12}"
            f"{expected[0]:<12}"
            f"{predicted:<12.4f}"
            f"{round(predicted)}"
        )


def main() -> None:
    print("Barebones Neural Network")
    print("Architecture: 2 inputs, 4 hidden neurons, 1 output")
    print()

    network = NeuralNetwork([2, 4, 1], seed=42)
    show_predictions(network, "Predictions before training:")
    print()
    network.train(XOR, epochs=5000, learning_rate=2.0, print_every=1000)
    print()
    show_predictions(network, "Predictions after training:")

    loss_start = network.loss_history[0]
    loss_end = network.loss_history[-1]
    print()
    print(
        f"Loss fell from {loss_start:.4f} to {loss_end:.6f} "
        f"over {len(network.loss_history)} epochs."
    )


if __name__ == "__main__":
    main()
