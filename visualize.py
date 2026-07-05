import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from network import Network


def compute_loss(net, data):        # Menghitung Mean Squared Error (MSE) untuk dataset
    total_loss = 0
    for x, y in data:
        output = net.feedforward(x)
        total_loss += np.sum((output - y) ** 2)
    return total_loss / len(data)


def compute_accuracy(net, test_data):
    correct = net.evaluate(test_data)
    return correct / len(test_data) * 100


print("Loading MNIST dataset")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def preprocess(x, y):
    data = []
    for img, label in zip(x, y):
        input_vec = img.reshape(784, 1) / 255.0
        target = np.zeros((10, 1))
        target[label] = 1.0
        data.append((input_vec, target))
    return data

training_data = preprocess(x_train, y_train)
test_data_raw = list(zip(
    [img.reshape(784, 1) / 255.0 for img in x_test],
    y_test
))

loss_sample = preprocess(x_test[:1000], y_test[:1000])
print("Preprocessing done!")

EPOCHS = 30
np.random.seed(42)
net = Network([784, 128, 64, 10])

loss_history = []
accuracy_history = []

print("Training with Visualization Tracking")
print("=" * 37)

for epoch in range(EPOCHS):
    import random
    random.shuffle(training_data)
    mini_batches = [
        training_data[k:k + 32]
        for k in range(0, len(training_data), 32)
    ]
    for mini_batch in mini_batches:
        net.update_mini_batch(mini_batch, 3.0)

    loss = compute_loss(net, loss_sample)       # Mencatat loss dan akurasi per epoch
    acc = compute_accuracy(net, test_data_raw)
    loss_history.append(loss)
    accuracy_history.append(acc)

    print(f"Epoch {epoch + 1:>2}/{EPOCHS} | Loss: {loss:.4f} | Accuracy: {acc:.2f}%")

print("=" * 45)


#  Plot 1: loss per epoch
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), loss_history, color="crimson", linewidth=2, marker="o", markersize=5)
plt.title("Loss per Epoch", fontsize=14, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.xticks(range(1, EPOCHS + 1))
plt.grid(True, linestyle="--", alpha=0.6)


#  Plot 2: akurasi per epoch
plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), accuracy_history, color="steelblue", linewidth=2, marker="o", markersize=5)
plt.title("Accuracy per Epoch", fontsize=14, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.xticks(range(1, EPOCHS + 1))
plt.ylim(0, 100)
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("loss_accuracy.png", dpi=100)
plt.show()
print("Saved: loss_accuracy.png")


#  Plot 3: visualisasi bobot layer pertama
print("\nVisualizing weights of the first hidden layer")
weights_layer1 = net.weights[0]  # Shape: (128, 784)
num_neurons = 64
grid_size = 8

fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
fig.suptitle("Weights Visualization — First Hidden Layer (64 neurons)", fontsize=12, fontweight="bold")

for i, ax in enumerate(axes.flatten()):
    if i < num_neurons:
        weight_img = weights_layer1[i].reshape(28, 28)
        ax.imshow(weight_img, cmap="RdBu", interpolation="nearest")
    ax.axis("off")

plt.tight_layout()
plt.savefig("weights_visualization.png", dpi=100)
plt.show()
print("Saved: weights_visualization.png")