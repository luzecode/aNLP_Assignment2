import torch
from utils import char_tensor
import math
import string as string_module


def compute_bpc(model, string):
    """
    Given a model and a string of characters, compute bits per character
    (BPC) using that model.

    Args:
        model: RNN-based model (RNN, LSTM, GRU, etc.)
        string: string of characters

    Returns:
        BPC for that set of string.
    """
    ################# STUDENT SOLUTION ################################
    model.eval() #setting model to evaluation mode disables dropout and batch norm. No random training

    #inititalize hidden and cell states to zeros, updated during training. Shape (model.py): [n_layers, batch_size=1, hidden_size]
    hidden, cell = model.init_hidden()

    # string.printable shows all ASCII printable characters
    all_characters = string_module.printable
    
    # summing up loss for each character prediction
    loss = 0

    #for char in range of the string we predict i+1 based on i
    for i in range(len(string)-1):
        curr_char = string[i] #current character
        index_char = all_characters.index(curr_char) #map the curr_char to an index
        input_tensor = torch.tensor([index_char]).long() # using the index to convert the char to a tensor of shape [1] tensor (scalar)
        #get the actual i+1 next character (ground truth)
        next_char = string[i+1]
        next_index_char = all_characters.index(next_char) #map next char to its index
        target_tensor = torch.tensor([next_index_char].long()) #using next char index to convert to a tensor of shape [1]
        
        # forward pass through LSTM with input: current character index (tensor)
        # Process through embedding --> LSTM --> decoder
        # Returns:output: [1, vocab_size] logits for next character
        #         hidden, cell: updated states for next timestep
        output,(hidden ,cell) = model(input_tensor, hidden, cell)

        # compute cross-entropy loss (torch.nn.CrossEntropyLoss) for this prediction
        # the Loss formula: -log(softmax(output)[true_class]) measures how confident the model is about the true next char
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, target_tensor)

        # sum up loss
        # .item() converts tensor to Python float for accumulation
        total_loss += loss.item()

    # calculate average loss per character by dividing by number of predictions (len(string) - 1)
    num_chars = len(string) - 1
    average_loss = total_loss / num_chars
    
    # convert from nats (base e) to bits (base 2) with formula: bits = nats / ln(2)
    # ln(2) ≈ 0.693 (log_2(x) = log_e(x) / log_e(2) = ln(x) / ln(2))
    bpc = average_loss / math.log(2)
    
    # return BPC score
    return bpc
    ###################################################################

