import argparse
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score




import flwr as fl
import pandas as pd
import perc_utils as utils


if __name__ == "__main__":
    N_CLIENTS = 5

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition-id",
        type=int,
        choices=range(0, N_CLIENTS),
        required=True,
        help="Specifies the artificial data partition",
    )
    args = parser.parse_args()
    partition_id = args.partition_id

    df = pd.read_csv('ready_to_train.csv')
    # Target Variable (died_at_the_hospital)
    HOSP_MORT = df['died_at_the_hospital'].values
    # Prediction Features
    features = df.drop(columns=['died_at_the_hospital'])    
    # Split into training set 80% and test set 20%
    X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    HOSP_MORT, 
                                                    test_size = .40, 
                                                    random_state = 42)

    # # Load the partition data
    # fds = FederatedDataset(dataset="hitorilabs/iris", partitioners={"train": N_CLIENTS})

    # dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
    # X = dataset[["petal_length", "petal_width", "sepal_length", "sepal_width"]]
    # y = dataset["species"]
    #unique_labels = df.load_split("train").unique("died_at_the_hospital")
    # # Split the on edge data: 80% train, 20% test
    # X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    # y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]

    

    model = Perceptron(max_iter=1000, tol=1e-4, eta0=1e-6, random_state=42, alpha=0.1) 

    def eval_learning(y_test, y_pred):
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(
            y_test, y_pred, average="weighted"
        )  # average argument required for multi-class
        prec = precision_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        return acc, rec, prec, f1

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model, n_features=X_train.shape[1], n_classes=3)

    # Define Flower client
    class GuardianClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            accuracy = model.score(X_train, y_train)
            return (
                utils.get_model_parameters(model),
                len(X_train),
                {"train_accuracy": accuracy},
            )

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict(X_test))
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            #y_pred = np.argmax(y_pred, axis=1).reshape(-1, 1)
            acc, rec, prec, f1 = eval_learning(y_test, y_pred)
            output_dict = {
                "loss": loss,
                "accuracy": accuracy,  # accuracy from tensorflow model.evaluate
                "acc": acc,
                "rec": rec,
                "prec": prec,
                "f1": f1,
            }
            return loss, len(X_test), output_dict

    # Start Flower client
    fl.client.start_client(
        server_address="0.0.0.0:8080", client=GuardianClient().to_client()
    )