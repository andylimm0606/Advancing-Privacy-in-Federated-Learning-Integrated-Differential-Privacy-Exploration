import flwr as fl
import argparse
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

def create_model():
    return SVC(kernel='linear')

class Client(fl.client.NumPyClient):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = create_model()
        print("Client initialized")

    def get_parameters(self, *args, **kwargs):
        print("Getting parameters")
        return self.model.get_params()

    def set_parameters(self, parameters):
        print("Setting parameters")
        print(parameters)
        
        self.model.set_params(**parameters)


    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
        return self.get_parameters(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = self.model.score(self.X_train, self.y_train)
        return float(accuracy), len(self.X_train), {}

N_CLIENTS = 3

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
HOSP_MORT = df['died_at_the_hospital'].values
features = df.drop(columns=['died_at_the_hospital'])

X_train, X_test, y_train, y_test = train_test_split(features, HOSP_MORT, test_size=0.20, random_state=0)

client = Client(X_train, y_train)

print("Starting client...")
fl.client.start_client(server_address="0.0.0.0:8080", client=client)
