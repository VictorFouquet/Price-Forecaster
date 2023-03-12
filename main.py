import json
import sys

from src.price_forecaster import PriceForecaster


def main():
    """
    Trains and tests a linear regression model then displays the resulting data
    """
    # Loads pretrained model if provided in command-line
    pre_trained = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        with open(model_path, "r") as f:
            pre_trained = json.load(f)
    
    # Loads config
    if pre_trained is not None:
        config = pre_trained["config"]
    else:
        with open("config.json", "r") as f:
            config = json.load(f)
            
    # Creates a price forecaster instance
    pf = PriceForecaster(config)
    # Trains price forecaster's model
    if pre_trained is not None:
        pf.train(pre_trained["model"])
    else:
        pf.train()
    # Tests price forecaster's model
    pf.test()
    # Saves price forecaster's model and stats
    pf.save_stats_and_model()
    # Displays test and training results
    pf.display()


if __name__ == "__main__":
    main()
