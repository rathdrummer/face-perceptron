# face-perceptron

Recognises 20x20 px images of emoticons based on values such as thoses described in the text files.
Using the current learning rate, it seems to peak at 83.7% accuracy reusing the training values.

Still to be implemented:
- Coformation to assignment standards: reading of filenames from command line arguments, output under certain format. It would be wise to create a second file to submit so that we can still use the 1st one for testing
- We could squeeze more accuracy out of the neural network if we preprocess the images; they are all randomly rotated off of 0°. The images could probably be righted by detecting the centre of gravity (as the top of the faces are a lot darker than the lower half) and rotating accordingly.
