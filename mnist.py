import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Tanh, Sigmoid, Softmax
from losses import binary_cross_entropy, binary_cross_entropy_prime, cross_entropy, cross_entropy_prime

def preprocess_data(x, y, limit):
    indices = []
    for i in range(10):
        indices.extend(np.where(y == i)[0][:limit])
    indices = np.random.permutation(indices)
    x, y = x[indices], y[indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y, num_classes = 10)
    y = y.reshape(len(y), 10, 1)
    return x, y

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

network = [
    Convolutional((1, 28, 28), 3, 32),
    Tanh(),
    Convolutional((32, 26, 26), 3, 64),
    Tanh(),
    Reshape((64, 24, 24), (64 * 24 * 24, 1)),
    Tanh(),
    Dense(64 * 24 * 24, 256),
    Tanh(),
    Dense(256, 128),
    Tanh(),
    Dense(128, 10),
    Softmax()
]

epochs = 30
learning_rate = 0.01
batch_size = 32

def get_batches(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

print("Training Start")
for e in range(epochs):
    print(f"Current Epoch: {e}")
    error = 0
    correct = 0
    for x_batch, y_batch in get_batches(x_train, y_train, batch_size):
        batch_error = 0
        batch_outputs = []
        for x, y in zip(x_batch, y_batch):
            output = x
            for layer in network:
                output = layer.forward(output)
            batch_outputs.append(output)
            batch_error += cross_entropy(y, output)
            if np.argmax(output) == np.argmax(y):
                correct += 1
        batch_error /= len(x_batch)
        error += batch_error
        for x, y in zip(x_batch, y_batch):
            output = x
            for layer in network:
                output = layer.forward(output)
            grad = cross_entropy_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
    error /= (len(x_train) / batch_size)
    accuracy = correct / len(x_train) * 100
    print(f"{e + 1}/{epochs}, error={error:.4f}, accuracy={accuracy:.2f}%")

correct = 0
for x, y in zip(x_test, y_test):
    output = x
    for layer in network:
        output = layer.forward(output)
    predicted = np.argmax(output)
    true = np.argmax(y)
    if predicted == true:
        correct += 1
    print(f"pred: {predicted}, true: {true}")

print(f"\nAccuracy: {correct/len(x_test)*100:.2f}%")