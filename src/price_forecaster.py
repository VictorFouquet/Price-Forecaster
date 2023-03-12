from datetime import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from src.data_processer import DataProcesser
from src.linear_regression import LinearRegression


class PriceForecaster:
    def __init__(self, config):
        self.config = config
        # Creates data processer
        self.data_processer = self.init_data_processer()
        # Sets train and test datasets
        [
        self.X_train,
        self.Y_train,
        self.X_test,
        self.Y_test
        ] = self.data_processer.build_sets()
        # Creates linear regression model
        features_len = self.data_processer.columns_len * self.config["period"]
        self.model = LinearRegression(features_len)
        # Data used to store test results
        self.accuracy = 0
        self.predictions = []
        self.targets = []
    
    def init_data_processer(self):
        """ Initializes the data processer """
        data_processer = DataProcesser(
            self.config["csv_path"], 
            self.config["total_time"],
            self.config["period"],
            self.config["target_label"],
            self.config["exclude_labels"]
        )
        data_processer.preprocess()
        return data_processer
    
    def train(self, model=None):
        """
        Trains the model.
            Args:
                model(dictionnary): Optionnal - Pre-trained model
        """
        # Initializes the model's weights and bias
        if model is not None:
            initial_w = np.array(model["weights"])
            initial_b = model["bias"]
        else:
            initial_w = np.random.rand(self.X_train.shape[1])
            initial_b = 0.
        # Trains the model using gradient descent 
        self.model.gradient_descent(
            self.X_train, self.Y_train, initial_w, initial_b,
            self.config["alpha"], self.config["iterations"],
            self.config["convergence_threshold"]
        )

    def test(self):
        """ Tests the model """
        # Resets accuracy
        self.accuracy = 0
        for i in range(self.X_test.shape[0]):
            # Makes prediction
            prediction = self.model.predict(self.model.weights, self.X_test[i], self.model.bias)
            # Scales result and target back to original scale
            scaled_prediction = prediction * self.data_processer.max_Y
            scaled_target = self.Y_test[i] * self.data_processer.max_Y
            # Stores result and target
            self.predictions.append(scaled_prediction)
            self.targets.append(scaled_target)
            # Updates accuracy
            diff = abs(scaled_prediction - scaled_target)
            acc = 100 - diff * 100 / scaled_target
            self.accuracy += acc
        # Computes average accuracy
        self.accuracy /= self.X_test.shape[0] - 1

    def save_stats_and_model(self, out_dir="models"):
        """
        Saves statistics and model to JSON file
            Args:
                out_dir(str): Optional - Output directory to store the JSON file
        """
        # Uses timestamp to name the output file
        file_name = f"model_{datetime.now().strftime('%Y_%m_%d_%H:%M:%S')}.json"
        out = os.path.join(out_dir, file_name)
        # Stores data in a dictionnary and dumps it to JSON string
        stats_and_model = json.dumps({
            "accuracy": self.accuracy,
            "model": {
                "weights": self.model.weights.tolist(),
                "bias": self.model.bias
            },
            "config": self.config
        }, indent=4)
        # Writes file to output folder
        with open(out, "w+") as f_output:
            f_output.write(stats_and_model)

    def display_training(self):
        """ Displays training statistics """
        # Creates cost and cost tail subplots
        cost_plot = plt.subplot2grid((3, 2), (0, 0))
        cost_tail_plot = plt.subplot2grid((3, 2), (0, 1))
        # Displays cost data
        cost_plot.plot(self.model.cost_history)
        cost_plot.set_title("Cost vs. iteration")
        cost_plot.set_ylabel('Cost')
        cost_plot.set_xlabel('iteration step')
        # Displays cost tail data
        cost_tail_plot.plot(100 + np.arange(len(self.model.cost_history[100:])), self.model.cost_history[100:])
        cost_tail_plot.set_title("Cost vs. iteration (tail)")
        cost_tail_plot.set_ylabel('Cost') 
        cost_tail_plot.set_xlabel('iteration step') 

    def display_test(self):
        """ Displays testing results """
        # Creates prediction plot
        prediction_plot = plt.subplot2grid((3, 2), (1, 0), 2, 2)
        # Displays prediction and target data
        prediction_plot.plot(
            range(len(self.predictions)),
            [p * 10 for p in self.predictions],
            'r', label="Prediction"
        )
        prediction_plot.plot(
            range(len(self.predictions)),
            [p * 10 for p in self.targets],
            'b', label="Actual prices"
        )
        prediction_plot.set_title("S&P500 SPY")
        prediction_plot.set_ylabel('Price ($)')
        prediction_plot.set_xlabel('Days')
        prediction_plot.legend()
    
    def show_stats(self):
        """ Displays training and testing stats and show the result """
        # Displays the result data on subplots
        self.display_training()
        self.display_test()
        # Shows the result data
        plt.tight_layout()
        plt.show()
