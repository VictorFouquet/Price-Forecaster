import copy
import numpy as np

class LinearRegression:
    def __init__(self, n):
        """
        Creates a linear regression instance
            Args:
                n (int): number of features in the model
        """
        self.weights = np.zeros((n,)) # Weights of the model
        self.bias = None              # Bias of the model
        self.cost_history = []        # Historical costs of the model

    def gradient_descent(self, x, y, w_init, b_init, alpha, num_iters, conv_threshold=1e-8):
        """
        Performs a gradient descent to optimize the model's weights and bias
            Args:
                x (np.array((total_time, n))): Inputs for training
                y (np.array((total_time,))): Targets for training
                w_init (np.array(n,)): Initial weights
                b_init (float): Initial bias
                alpha (float): Learning rate
                num_iters(int): number of iterations to perform
            Returns
                weights (np.array((n,))): weights after training
                bias (float): bias after training
                cost_history (list(float)): cost records
        """
        # Sets initial weights and bias
        self.weights = copy.deepcopy(w_init)
        self.bias = b_init
        for i in range(num_iters):
            # Computes partial derivatives of J with respect to w and with respect to b
            dj_dw, dj_db = self.compute_gradient(self.weights, self.bias, x, y)
            # Updates weight and bias with the computed gradients
            self.weights = self.weights - alpha * dj_dw
            self.bias = self.bias - alpha * dj_db
            # Stores current cost
            self.cost_history.append( self.compute_cost(self.weights, self.bias, x, y) )
            # Displays cost to inform user about the progression
            if i% np.ceil(num_iters / 100) == 0:
                print(f"Iteration {i:4d}: Cost {self.cost_history[-1]:8.9f}")
            # Checks for convergence
            if (
                len(self.cost_history) > 1 and 
                abs(self.cost_history[-1] - self.cost_history[-2]) < conv_threshold
            ):
                return self.weights, self.bias, self.cost_history
        
        return self.weights, self.bias, self.cost_history

    def compute_gradient(self, w, b, x, y):
        """
        Computes the gradient for one set of weights and bias
            Args:
                w (np.array(n, )): current weights
                b (float): current bias
                x (np.array((m, n))): input features
                y (np.array((m,))): targets
            Returns
                dj_dw (np.array((n,))) Gradient of cost function with respect to w
                dj_db (float) Gradient of cost function with respect to b
        """
        m, n = x.shape
        dj_dw = np.zeros((n,))
        dj_db = 0.
        # Sums error for J with respect to w and with respect to b
        for i in range(m):
            # Computes the general error for given weights and bias
            err = (np.dot(x[i], w) + b) - y[i]
            # Updates gradient of J with respect to b
            dj_db += err
            # Updated gradient of J with respect to w
            for j in range(n):
                # Computes the error for every component of vector w
                dj_dw[j] += err * x[i, j]
        # Averages the error
        dj_dw /= m
        dj_db /= m

        return dj_dw, dj_db

    def compute_cost(self, w, b, x, y):
        """
        Computes the cost of the model using mean squared error
            Args:
                w (np.array(n,)): weights to evaluate
                b (float); bias to evaluate
                x (np.array((m, n))): inputs
                y (np.array(m)): outputs
            Returns
                total_cost: mean squared error of the function
        """
        total_cost = 0
        m = x.shape[0]
        # Computes the sum of squared errors
        for i in range(m):
            total_cost += pow(self.predict(w, x[i], b) - y[i], 2)
        # Averages error
        total_cost /= (2 * m)
        return total_cost

    def predict(self, w, x, b):
        """
        Makes a prediction, method used as the linear regression model
            Args:
                w (np.array((n,))): weights
                x (np.array((n,)): inputs
                b (float): bias
            Returns
                prediction of the model (float)
        """
        return np.dot(w, x) + b
