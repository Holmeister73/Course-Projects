Task: Predict the next word that fits the sentence after given first 3 words.

Training data: 	Input: 372k sequences of 3 numbers where the numbers are the places of the 3 words in dictionary (vocab.npy) 
			Target: 372k single numbers where the number is the place of the next word in the dictionary

Both validation data and test data are of 46500 length.

Model: Feed forward neural network with 1 hidden layer and 1 embedding layer. Embedding dimension is 256 and learned for each word. Then the embeddings of 3 words are concatenated. After that we have 1 hidden layer and 1 output layer. Output layer is a softmaw with dimension equal to dictionary's size.
Note: Libraries like pytorch and tensorflow weren't used and both forward and backward propagation is written from scratch.


Note: There is also tsne.py file you can run where you can see which words are closer according to the model. 