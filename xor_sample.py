import numpy as np
from network import Network

training_data = [
    (np.array([[0], [0]]), np.array([[0]])),
    (np.array([[0], [1]]), np.array([[1]])),
    (np.array([[1], [0]]), np.array([[1]])),
    (np.array([[1], [1]]), np.array([[0]])),
]

np.random.seed(42)
net = Network([2, 4, 1])    # 2 input, 4 hidden, 1 output

print(" XOR Neural Network - Training")
print("=" * 40)

net.train(
    training_data=training_data,
    epochs=10000,
    mini_batch_size=4,
    eta=1.0,
)

print()
print(" Hasil Prediksi Setelah Training")
print("-" * 35)
print(f"{'Input':<15} {'Target':<10} {'Prediksi':<12} {'Status'}")
print("-" * 45)

for x, y in training_data:
    pred = net.feedforward(x)[0][0]
    target = y[0][0]
    rounded = round(pred)
    status = "Benar" if rounded == target else "Salah"
    input_str = f"[{int(x[0][0])}, {int(x[1][0])}]"
    print(f"{input_str:<15} {int(target):<10} {pred:<12.6f} {status}")