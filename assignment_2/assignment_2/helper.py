from model.ffnn import compute_loss
import numpy as np


def batch_train(X, Y, model, train_flag=False):
    ################################# STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) Use your neural network to predict the intent
    #         (without any training) and calculate the accuracy 
    #         of the classifier. Should you be expecting high
    #         numbers yet?
    #         2) if train_flag is true, run the training for 1000 epochs using 
    #         learning rate = 0.005 and use this neural network to predict the 
    #         intent and calculate the accuracy of the classifier
    #         3) Then, plot the cost function for each iteration and
    #         compare the results after training with results before training
    epochs = 1000
    l_rate = 0.005
    M = X.shape[1]
    costs = []

    if train_flag:
        for e in range(epochs):
            P = model.forward(X)
    
            loss = compute_loss(P, Y)
            costs.append(loss)
        
            dW1, db1, dW2, db2 = model.backward(X, Y)
            
            model.W1 -= (l_rate / M) * dW1
            model.b1 -= (l_rate / M) * db1
            model.W2 -= (l_rate / M) * dW2
            model.b2 -= (l_rate / M) * db2
            
            if (e + 1) % 100 == 0:
                print(f"Epoch {e+1}, Loss: {loss:.4f}")
        
    
        prediction = model.predict(X, one_hot=True)
        accuracy = np.mean(np.argmax(prediction, axis=0) == np.argmax(Y, axis=0))
        print(f"Accuracy after: {accuracy:.4f}")
    
    return costs
    ###############################################################################


def minibatch_train(X, Y, model, train_flag=False, batch_size = 64):
    ########################## STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) As bonus, train your neural network with batch size = 64
    #         and SGD (batch size = 1) for 1000 epochs using learning rate
    #         = 0.005. Then, plot the cost vs iteration for both cases.
    epochs = 1000
    l_rate = 0.005
    M = X.shape[1]
    costs = []

    
    if train_flag:
        for e in range(epochs):
            indices = np.random.permutation(M)
            X_shuffled = X[:, indices]
            Y_shuffled = Y[:, indices]
            
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, M, batch_size):
                X_batch = X_shuffled[:, i:i+batch_size]
                Y_batch = Y_shuffled[:, i:i+batch_size]
                batch_m = X_batch.shape[1]
                
                P = model.forward(X_batch)
                loss = compute_loss(P, Y_batch)
                epoch_loss += loss
                num_batches += 1
                
                dW1, db1, dW2, db2 = model.backward(X_batch, Y_batch)
                
                model.W1 -= (l_rate / batch_m) * dW1
                model.b1 -= (l_rate / batch_m) * db1
                model.W2 -= (l_rate / batch_m) * dW2
                model.b2 -= (l_rate / batch_m) * db2
            
            costs.append(epoch_loss / num_batches)
            
            if (e + 1) % 100 == 0:
                print(f"Epoch {e+1}, Loss: {costs[-1]:.4f}")
            
        prediction = model.predict(X, one_hot=True)
        accuracy = np.mean(np.argmax(prediction, axis=0) == np.argmax(Y, axis=0))
        print(f"Accuracy after training: {accuracy:.4f}")
    
    return costs
    #########################################################################
