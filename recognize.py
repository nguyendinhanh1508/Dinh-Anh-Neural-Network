import numpy as np
from PIL import Image
from storage import param_storage
from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Tanh, Softmax

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float32)
    img_array = (255 - img_array) / 255.0
    return img_array

def build_network():
    return [
        Convolutional((1, 28, 28), 3, 5),
        Tanh(),
        Reshape((5, 26, 26), (5 * 26 * 26, 1)),
        Dense(5 * 26 * 26, 100),
        Tanh(),
        Dense(100, 10),
        Softmax()
    ]

def predict_digit(image_path):
    img = preprocess_image(image_path)
    network = build_network()
    if param_storage.load():
        for layer in network:
            if hasattr(layer, 'load_weights'):
                layer.load_weights()
    output = img[np.newaxis, :, :]
    for layer in network:
        output = layer.forward(output)
    digit = np.argmax(output)
    confidence = np.max(output)
    return digit, confidence

if __name__ == "__main__":
    image_path = input("Enter path to your image: ").strip('"')
    try:
        digit, confidence = predict_digit(image_path)
        print(f"Predicted digit: {digit} with {confidence*100:.2f}% confidence")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your image is a clear 28x28 grayscale digit image")