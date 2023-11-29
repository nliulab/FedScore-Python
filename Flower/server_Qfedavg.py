import warnings
warnings.simplefilter('ignore')
import json
import flwr as fl
import utils
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from typing import Dict


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}

def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""
    # get input from csv:
    directory = '../output/client01_data'
    file_name = "design_matrix.csv"

    X_train, y_train, num_features = utils.get_dat_csv(file_name, directory, variable_list=sys.argv[2].split(','))
    X_test, y_test = X_train, y_train

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        print(utils.get_model_parameters(model))
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for K rounds of federated learning
if __name__ == "__main__":
    min_clients = int(sys.argv[1])
    variable_list = sys.argv[2].split(',')
    with open('../output/server_data/global_cut_vector.json') as f:
        cut_off = json.loads(f.read())
    with open('../output/server_data/combined_categorical_vars.json') as f:
        combined_categories = json.loads(f.read())
    n_features = 0
    for var in variable_list:
        if var in cut_off:
            n_features += len(cut_off[var])
        elif var in combined_categories:
            n_features += len(combined_categories[var]) - 1
    model = LogisticRegression(penalty=None, solver='newton-cg', max_iter=1000)
    utils.set_initial_params(model, n_classes=2, n_features=n_features)
    strategy = fl.server.strategy.QFedAvg(
        fraction_evaluate=1.0,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        min_fit_clients=min_clients,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
        initial_parameters=fl.common.ndarrays_to_parameters(list(utils.get_model_parameters(model))),
        q_param=0.5,
        qffl_learning_rate=0.5,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=30),
    )
