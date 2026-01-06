import torch
import torch.nn as nn


# Here is a pseudocode to help with your LSTM implementation. 
# You can add new methods and/or change the signature (i.e., the input parameters) of the methods.
class LSTM(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, layer_size):
        """Think about which (hyper-)parameters your model needs; i.e., parameters that determine the
        exact shape (as opposed to the architecture) of the model. There's an embedding layer, which needs 
        to know how many elements it needs to embed, and into vectors of what size. There's a recurrent layer,
        which needs to know the size of its input (coming from the embedding layer). PyTorch also makes
        it easy to create a stack of such layers in one command; the size of the stack can be given
        here. Finally, the output of the recurrent layer(s) needs to be projected again into a vector
        of a specified size."""
        ############################ STUDENT SOLUTION #############################
        #call the parent initializer first
        super(LSTM, self).__init__()
        
        # hyperparameters:
        self.in_size = in_size   #input size: n unique characters
        self.out_size = out_size #output size: predict one of n characters
        self.hidden_size = hidden_size #dimensions of hidden states
        self.layer_size = layer_size #amount of layers

        # self.condoer: encodes characters into vectors with input size or vocabulary size and hidden size to determine how many dimensions the vector should have
        self.encoder = nn.Embedding(self.in_size, self.hidden_size) 
        #self.lstm: with hidden_size as input (output from the encoder) and hidden_size as the internal state size. Numbers of layers for a stacked LSTM
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.layer_size)
        #self.decoder: from hidden states formulates into output layer. Input is the hidden size from the layers before and output is the output size, 
        # which will be a probability distribution over the vocabulary
        self.decoder = nn.Linear(self.hidden_size, self.out_size)
        ##########################################################################

    def forward(self, input,hidden_cell):
        """Your implementation should accept input character, hidden and cell state,
        and output the next character distribution and the updated hidden and cell state."""
        ############################ STUDENT SOLUTION ############################
        batch_size = 1 #as we predict only one character at a time
        hidden, cell = hidden_cell
        #first encode the input characters,input shape: scalar or one-dimensional tensor -> encoded shape: [1, hidden_size]
        encoded = self.encoder(input.view(1, -1))

        # then pass it to the lstm with hidden and cel state
        # input shape for LSTM/ output from encoding: [1, 1, hidden_size]
        # cell and hidden shape: [layer_size, batch_size, hidden_size]
        output, (hidden, cell) = self.lstm(encoded.view(1, batch_size, -1), (hidden, cell))

        # then reshape output and pass through decoder
        # input shape for decoder/ output shape from LSTM: [1, 1, hidden_size]
        # output shape after reshape: [1, hidden_size] and decoder output shape: [1, out_size]
        output = self.decoder(output.view(batch_size, -1))
        
        return output, (hidden, cell)

        ##########################################################################

    def init_hidden(self):
        """Finally, you need to initialize the (actual) parameters of the model (the weight
        tensors) with the correct shapes."""
        ############################ STUDENT SOLUTION ############################
        # fill in the hidden sate with zeros with shape: [layer_size, batch_size (1), hidden_size]
        hidden = torch.zeros(self.layer_size, 1, self.hidden_size)
        
        # initialize cell state with zeros with shape: [layer_size, batch_size (1), hidden_size]
        # Cell state has the same shape as hidden state
        cell = torch.zeros(self.layer_size, 1, self.hidden_size)
        
        # return both states as a tuple
        return hidden, cell
        ##########################################################################
