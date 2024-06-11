from typing import List, Tuple, Dict

import numpy as np
from sklearn.linear_model import Perceptron

from flwr.common import NDArrays, Metrics, Scalar


def get_model_parameters(model: Perceptron) -> NDArrays:
    """Return the parameters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params

def set_model_params(model: Perceptron, params: NDArrays) -> Perceptron:
    """Set the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: Perceptron, n_classes: int, n_features: int):
    """Set initial parameters as zeros.

    Required since model params are uninitialized until model.fit is called but server
    asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Scalar]:
    """Compute weighted average.

    It is generic implementation that averages only over floats and ints and drops the
    other data types of the Metrics.
    """
    print(metrics)
    # num_samples_list can represent number of sample or batches depending on the client
    num_samples_list = [n_batches for n_batches, _ in metrics]
    num_samples_sum = sum(num_samples_list)
    metrics_lists: Dict[str, List[float]] = {}
    for num_samples, all_metrics_dict in metrics:
        #  Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            if isinstance(value, (float, int)):
                metrics_lists[single_metric] = []
        # Just one iteration needed to initialize the keywords
        break

    for num_samples, all_metrics_dict in metrics:
        # Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            # Add weighted metric
            if isinstance(value, (float, int)):
                metrics_lists[single_metric].append(float(num_samples * value))

    weighted_metrics: Dict[str, Scalar] = {}
    for metric_name, metric_values in metrics_lists.items():
        weighted_metrics[metric_name] = sum(metric_values) / num_samples_sum

    return weighted_metrics



# def load_admit():
#     """
#     Loads the data from local.
#     """

#     df = pd.read_csv('ready_to_train.csv')
#     # Target Variable (died_at_the_hospital)
#     HOSP_MORT = df['died_at_the_hospital'].values
#     # Prediction Features
#     features = df.drop(columns=['died_at_the_hospital'])    
#     # Split into training set 80% and test set 20%
#     x_train, x_test, y_train, y_test = train_test_split(features, 
#                                                     HOSP_MORT, 
#                                                     test_size = .20, 
#                                                     random_state = 0)
#     return (x_train, y_train), (x_test, y_test)
