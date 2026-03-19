import random
import numpy as np
import pickle


class Network(object):
    """
    Implementasi Feedforward Neuralnetwork dengan algoritma backpropagation,
    menggunakan Mini-batch Stochastic Gradient Descent (SGD) untuk training.
    """

    def __init__(self, sizes):      # inisialisasi jaringan saraf
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [
            np.random.randn(y, x)
            for x, y in zip(sizes[:-1], sizes[1:])
        ]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def train(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Parameters:
            training_data (list): list tuple (x, y) data latih.
            epochs (int): jumlah epoch training.
            mini_batch_size (int): ukuran setiap mini-batch.
            eta (float): learning rate.
            test_data (list): data uji untuk evaluasi per epoch.
        """
        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                n_test = len(test_data)
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [
            w - (eta / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb
            for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # forward pass
        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass — output layer
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # backward pass — hidden layer
        for i in range(2, self.num_layers):
            z = zs[-i]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-i + 1].T, delta) * sp
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activations[-i - 1].T)

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [
            (np.argmax(self.feedforward(x)), y)
            for (x, y) in test_data
        ]
        return sum(int(pred == y) for pred, y in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def save(self, filename):   # simpan model ke file menggunakan pickle
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases]
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)

        net = Network(data["sizes"])
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net


def sigmoid(z):     # fungsi aktivasi sigmoid
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):   # turunan dari fungsi sigmoid
    return sigmoid(z) * (1 - sigmoid(z))


if __name__ == "__main__":
    net = Network([2, 3, 1])
    x = np.array([[0.5], [0.1]])
    y = net.feedforward(x)
    print("Output jaringan:", y)
