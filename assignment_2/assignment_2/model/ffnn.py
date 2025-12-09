import numpy as np
import numpy.typing as npt

from model.model_utils import softmax, relu, relu_prime
from typing import Tuple


class NeuralNetwork(object):
    def __init__(
        self, 
        input_size: int,
        hidden_size: int, 
        num_classes: int,
        seed: int = 1
    ):
        """
        Initialize neural network's weights and biases.
        """
        ############################# STUDENT SOLUTION ####################
        # YOUR CODE HERE
        #     TODO:
        #         1) Set a seed so that your model is reproducible
        #         2) Initialize weight matrices and biases with uniform
        #         distribution in the range (-1, 1).
        self.hidden_size = 150
        np.random.seed(seed)

        self.W1 = np.random.uniform(-1.0, 1.0, size=(hidden_size, input_size))
        self.b1 = np.random.uniform(-1.0, 1.0, size=(hidden_size, 1))

        self.W2 = np.random.uniform(-1.0, 1.0, size=(num_classes, hidden_size))
        self.b2 = np.random.uniform(-1.0, 1.0, size=(num_classes, 1))
        ###################################################################

    def forward(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Forward pass with X as input matrix, returning the model prediction
        Y_hat.
        """
        ######################### STUDENT SOLUTION #########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Perform only a forward pass with X as input.
        Z1 = self.W1 @ X + self.b1
        A1 = relu(Z1)            

        Z2 = self.W2 @ A1 + self.b2 
        Y_hat = softmax(Z2)        

        return Y_hat
        #####################################################################

    def predict(self, X: npt.ArrayLike, one_hot = False) -> npt.ArrayLike:
        """
        Create a prediction matrix with `self.forward()`
        """
        ######################### STUDENT SOLUTION ###########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Create a prediction matrix of the intent data using
        #         `self.forward()` function. The shape of prediction matrix
        #         should be similar to label matrix produced with
        #         `labels_matrix()`
        P = self.forward(X)

        if one_hot:
            K, M = P.shape
            preds = np.zeros((K, M), dtype=P.dtype)
            idx = np.argmax(P, axis=0)
            preds[idx, np.arange(M)] = 1
            return preds

        return P
        ######################################################################

    def backward(
        self, 
        X: npt.ArrayLike, 
        Y: npt.ArrayLike
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """
        Backpropagation algorithm.
        """
        ########################## STUDENT SOLUTION ###########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Perform forward pass, then backpropagation
        #         to get gradient for weight matrices and biases
        #         2) Return the gradient for weight matrices and biases
        P = self.forward(X) 

        Z1 = self.W1 @ X + self.b1          
        A1 = relu(Z1)    

        dZ2 = P - Y                        
        dW2 = dZ2 @ A1.T                    
        db2 = np.sum(dZ2, axis=1, keepdims=True) 

        dA1 = self.W2.T @ dZ2               
        dZ1 = dA1 * relu_prime(Z1)          
        dW1 = dZ1 @ X.T                    
        db1 = np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2


        #######################################################################

def compute_loss(pred: npt.ArrayLike, truth: npt.ArrayLike) -> float:
    """
    Compute the cross entropy loss.
    """
    ########################## STUDENT SOLUTION ###########################
    # YOUR CODE HERE
    #     TODO:
    #         1) Compute the cross entropy loss between your model prediction
    #         and the ground truth.

    val = 0.0000000001
    L = -np.sum(truth * np.log(pred + val), axis=0)
    return float(np.mean(L))  
    #######################################################################