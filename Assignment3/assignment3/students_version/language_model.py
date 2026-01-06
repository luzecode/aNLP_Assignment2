import torch
import torch.nn as nn
import string
import time
import unidecode
import matplotlib.pyplot as plt

from utils import char_tensor, random_training_set, time_since, random_chunk, CHUNK_LEN
from evaluation import compute_bpc
from model.model import LSTM


def generate(decoder, prime_str='A', predict_len=100, temperature=0.8):
    hidden, cell = decoder.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str
    all_characters = string.printable

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, (hidden, cell) = decoder(prime_input[p], (hidden, cell)) 
    inp = prime_input[-1]

    for p in range(predict_len):
        output, (hidden, cell) = decoder(inp, (hidden, cell))

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    return predicted


def train(decoder, decoder_optimizer, inp, target):
    hidden, cell = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0
    criterion = nn.CrossEntropyLoss()

    for c in range(CHUNK_LEN):
        output, (hidden, cell) = decoder(inp[c], (hidden, cell))
        loss += criterion(output, target[c].view(1))

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / CHUNK_LEN


def tuner(n_epochs=3000, print_every=100, plot_every=10, hidden_size=128, n_layers=2,
          lr=0.005, start_string='A', prediction_length=100, temperature=0.8):
        # YOUR CODE HERE
        #     TODO:
        #         1) Implement a `tuner` that wraps over the training process (i.e. part
        #            of code that is ran with `default_train` flag) where you can
        #            adjust the hyperparameters
        #         2) This tuner will be used for `custom_train`, `plot_loss`, and
        #            `diff_temp` functions, so it should also accomodate function needed by
        #            those function (e.g. returning trained model to compute BPC and
        #            losses for plotting purpose).

        ################################### STUDENT SOLUTION #######################
        # getting the vocab size from all printable characters
        all_characters = string.printable
        n_characters = len(all_characters)
        
        # create the LSTM model with specified hyperparameters (input_size, output_size, hidden_size, n_layers)
        decoder = LSTM(n_characters, n_characters, hidden_size, n_layers)
        
        # optimizer with specified learning rate
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
        
        # variable that I want to track
        start = time.time()           # elapsed time calculation
        all_losses = []               # collect averaged losses for plotting
        loss_avg = 0                  # sum up loss
        
        # training loop over n epochs
        for epoch in range(1, n_epochs + 1):
            # using the function from utils to get random training pair (input chars, target chars)
            input, target = random_training_set()
            
            # one training step and get loss
            loss = train(decoder, decoder_optimizer, input, target)
            
            # add to overal loss
            loss_avg += loss
            
            # print status and generate sample
            if epoch % print_every == 0:
                print('[{} ({} {}%) {:.4f}]'.format( # [elapsed_time (epoch progress%) current_loss]
                    time_since(start), 
                    epoch, 
                    epoch / n_epochs * 100, 
                    loss))
                
                # with generate() create a text from sample text
                print(generate(decoder, start_string, prediction_length, temperature))
                print()
            
            # track averaged loss for plotting
            if epoch % plot_every == 0:
                # calculate average loss over plot_every epochs
                avg = loss_avg / plot_every
                all_losses.append(avg)
                # set loss to 0 again
                loss_avg = 0
        
            # return trained model and loss history
        return decoder, all_losses
        ############################################################################

def plot_loss(lr_list):
    # YOUR CODE HERE
    #     TODO:
    #         1) Using `tuner()` function, train X models where X is len(lr_list),
    #         and plot the training loss of each model on the same graph.
    #         2) Don't forget to add an entry for each experiment to the legend of the graph.
    #         Each graph should contain no more than 10 experiments.
    ###################################### STUDENT SOLUTION ##########################
    # create figure
    plt.figure(figsize=(10, 6))
    
    # train and plot for each learning rate
    for lr in lr_list[:10]:
        print(f"Training with lr={lr}...")
        model, losses = tuner(n_epochs=1000, lr=lr, print_every=1000)
        plt.plot(losses, label=f'lr={lr}')
    
    # other info for plot
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss for Different Learning Rates')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_comparison.png')
    plt.show()
    ##################################################################################

def diff_temp(temp_list):
    # YOUR CODE HERE
    #     TODO:
    #         1) Using `tuner()` function, try to generate strings by using different temperature
    #         from `temp_list`.
    #         2) In order to do this, create chunks from the test set (with 200 characters length)
    #         and take first 10 characters of a randomly chosen chunk as a priming string.
    #         3) What happen with the output when you increase or decrease the temperature?
    ################################ STUDENT SOLUTION ################################
    # train a model
    model, losses = tuner(n_epochs=2000, print_every=500)
    
    # random chunk and prime string
    chunk = random_chunk()
    prime_str = chunk[:10]
    
    print(f"Prime string: '{prime_str}'\n")
    
    # generate text with different temperatures
    for temp in temp_list:
        print(f"Temperature {temp}:")
        generated = generate(model, prime_str, 100, temp)
        print(generated)
        print()
    ##################################################################################

def custom_train(hyperparam_list):
    """
    Train model with X different set of hyperparameters, where X is 
    len(hyperparam_list).

    Args:
        hyperparam_list: list of dict of hyperparameter settings

    Returns:
        bpc_dict: dict of bpc score for each set of hyperparameters.
    """
    TEST_PATH = './data/dickens_test.txt'
    string = unidecode.unidecode(open(TEST_PATH, 'r').read())
    # YOUR CODE HERE
    #     TODO:
    #         1) Using `tuner()` function, train X models with different
    #         set of hyperparameters and compute their BPC scores on the test set.

    ################################# STUDENT SOLUTION ##########################
    bpc_dict = {}
    
    # train model with different parameters
    for i, params in enumerate(hyperparam_list):
        print(f"\nModel {i+1}/{len(hyperparam_list)}: {params}")
        
        # these parameters
        model, losses = tuner( n_epochs=params.get('n_epochs', 1000),hidden_size=params.get('hidden_size', 128),
            n_layers=params.get('n_layers', 2),lr=params.get('lr', 0.005), print_every=params.get('print_every', 500))
        
        # compute BPC
        bpc = compute_bpc(model, string)
        
        # store result
        config_name = f"h={params.get('hidden_size', 128)}, l={params.get('n_layers', 2)}, lr={params.get('lr', 0.005)}"
        bpc_dict[config_name] = bpc
        print(f"BPC: {bpc:.4f}")

    
    return bpc_dict
    #############################################################################