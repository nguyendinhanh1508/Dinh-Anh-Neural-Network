# Dinh Anh's Neural network
This is Dinh Anh's own AI that is trained on the MNIST dataset<br>
This AI has a 85% accuracy after around 20 epochs of training, it will improve in the future<br>
I used a Convolutional Neural Network for this to work, plus it's built almost entirely using numpy, for the convolutional layer, activation functions, or dense layers.<br>
The training data is already stored in the network_state.npz, so when you download this you can test it right away<br>

# How it works

1. Layers: <br>
- Input: 28x28 grayscale image.
- Convolutional: 5 filters (3x3 kernel), Tanh activation.
- Reshape: Flatten for Dense layers.
- Dense(100): Fully connected layer, Tanh activation.
- Dense(10): Output layer, Softmax activation.
2. Training process (```train.py```):
- Uses Keras to fetch the MNIST dataset.
- Normalizes pixel value to [0, 1] and reshapes images to (1, 28, 28);
3. Forward pass:
- **Convolutional** layer extracts features.
- **Tanh Activation** introduces non-linearity for more complex patterns.
- **Reshape** prepares data for dense layers.
- Two **Dense** layers refine predictions.
- **Softmax** converts outputs into probability.
4. Backward pass:
- Computes gradient using cross-entropy loss.
- Updates weights via Gradient Descent (learning rate = 0.01).
5. Storing data:
- Weights and biases are stored in network_state.npz.
6. Predictions:
- Resizes the image to 28x28, converts to grayscale, inverts colors, and normalize to [0, 1];
- Forward pass passes the image through the treained network and return the digit with highet probability.

# Structure

1. ```activations.py``` & ```activation.py```: 2D Convolutional layer implementation..
2. ```dense.py```: Fully connected (Dense) layer.
3. ```layer.py```: Layer class.
4. ```losses.py```: MSE, Cross-Entropy loss functions.
5. ```network.py```: Helper functions for training.
6. ```recognize.py```: Code for recognizing numbers.
7. ```reshape.py```: Prepare data for Dense layers.
8. ```storage.py```: Storing data in ```network_state.npz``` implementation.
9. ```test.py```: For checking the accuracy of the neural network.
10. ```train.py```: For training the neural network.

# How to use
Download the dependencies in ```requirements.txt```<br>
To train from scratch, you can delete the network_state.npz and run the file train.py to let it train<br>
You can also change the number of epochs and learning rate, but bear in mind, the faster it learns, the less improvement you will get from each epoch<br>
If not then this network is already trained over around 500 epochs and at this point the momentum has already reduced to really small, only being able to improve a fraction of the loss, hence there's no point in training it, so you can test it right away<br>
To test this AI, simply run the test.py file<br>
To recognize an image, simply drop an image into this folder and run recognize.py, just input the path to the image and it will output the most possible number.
# Credits
Credits to the Independent Code for teaching me about the basics of the network<br>
