# Dinh Anh's Neural network
This is Dinh Anh's own AI that is trained on the MNIST dataset<br>
This AI has a 85% accuracy after around 20 epochs of training, it will improve in the future<br>
I used a Convolutional Neural Network for this to work, plus it's built almost entirely using numpy, for the convolutional layer, activation functions, or dense layers.<br>
The training data is already stored in the network_state.npz, so when you download this you can test it right away<br>
# How to use
To train from scratch, you can delete the network_state.npz and let it train<br>
If not then this network is already trained over around 500 epochs and at this point the momentum has already reduced to really small, only being able to improve a fraction of the loss, hence there's no point in training it, so you can test it right away<br>
To test this AI, simply run the test.py file<br>
I haven't added testing on actual images yet but I will work on it in the future<br>
# Credits
Credits to the Independent Code for teaching me about the basics of the network<br>
