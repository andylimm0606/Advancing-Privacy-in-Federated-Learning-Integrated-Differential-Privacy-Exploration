import flwr as fl
import perc_utils as utils
from sklearn.linear_model import Perceptron

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds, one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}

# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = Perceptron()
    utils.set_initial_params(model, n_classes=3, n_features=37)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=3,
        fit_metrics_aggregation_fn=utils.weighted_average,
        evaluate_metrics_aggregation_fn=utils.weighted_average,
        on_evaluate_config_fn=evaluate_config,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=25),
    )