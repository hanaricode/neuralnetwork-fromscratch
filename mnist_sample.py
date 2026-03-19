import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from network import Network

#  Load Dataset MNIST
print("   MNIST Neural Network - Loading Data")
print("=" * 35)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Data latih : {x_train.shape[0]} gambar")
print(f"Data uji   : {x_test.shape[0]} gambar")
print(f"Ukuran tiap gambar: {x_train.shape[1]}x{x_train.shape[2]} pixel")

#  Preprocessing Data
#  Flatten gambar 28x28 -> vektor 784
#  Normalisasi pixel 0-255 -> 0.0-1.0
#  Label -> one-hot encoding
def preprocess(x, y):
    data = []
    for img, label in zip(x, y):
        input_vec = img.reshape(784, 1) / 255.0
        target = np.zeros((10, 1))
        target[label] = 1.0
        data.append((input_vec, target))
    return data

print("\nMemproses data")
training_data = preprocess(x_train, y_train)
test_data_raw = list(zip(
    [img.reshape(784, 1) / 255.0 for img in x_test],
    y_test))
print("Preprocessing selesai!")

#  Visualisasi Contoh Data
print("\nMenunjukkan contoh gambar dari dataset")
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Contoh gambar MNIST", fontsize=12, fontweight="bold")

for i, ax in enumerate(axes.flatten()):
    ax.imshow(x_train[i], cmap="gray")
    ax.set_title(f"Label: {y_train[i]}", fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("mnist_samples.png", dpi=100)
plt.show()
print("Gambar contoh di simpan sebagai 'mnist_samples.png'")


print(" Memulai Training")
print("=" * 25)
print("Arsitektur : [784 → 64 → 10]")
print("Epochs     : 10")
print("Batch size : 32")
print("Eta        : 3.0")
print("-" * 30)

np.random.seed(42)
net = Network([784, 128, 64, 10])

net.train(
    training_data=training_data,
    epochs=30,
    mini_batch_size=32,
    eta=3.0,
    test_data=test_data_raw)

# Evaluasi Akhir
correct = net.evaluate(test_data_raw)
total = len(test_data_raw)
accuracy = correct / total * 100

print("  Hasil Evaluasi Akhir")
print("=" * 25)
print(f"Benar  : {correct} / {total}")
print(f"Akurasi: {accuracy:.2f}%")

#  Visualisasi Prediksi
print("\nMenunjukkan contoh prediksi")
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle(f"Contoh prediksi (Akurasi: {accuracy:.2f}%)", fontsize=14, fontweight="bold")

for i, ax in enumerate(axes.flatten()):
    img = x_test[i]
    input_vec = img.reshape(784, 1) / 255.0
    pred = np.argmax(net.feedforward(input_vec))
    actual = y_test[i]

    ax.imshow(img, cmap="gray")
    color = "green" if pred == actual else "red"
    ax.set_title(f"Pred: {pred} | Asli: {actual}", fontsize=9, color=color)
    ax.axis("off")

plt.tight_layout()
plt.savefig("mnist_predictions.png", dpi=100)
plt.show()
print("Gambar prediksi disimpan sebagai 'mnist_predictions.png'")