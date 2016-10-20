# face-perceptron

Recognises 20x20 px images of emoticons based on values such as thoses described in the text files.
Using the current learning rate, it seems to peak at 83.7% accuracy reusing the training values.
Conforms to assignment standards: reads filenames from command line arguments, output under certain format.

Still to be implemented:
- We could squeeze more accuracy out of the neural network if we preprocess the images; they are all randomly rotated off of 0°. The images could probably be righted by detecting the centre of gravity (as the top of the faces are a lot darker than the lower half) and rotating accordingly.
