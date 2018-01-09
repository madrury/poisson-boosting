import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingPoissonRegressor:


    def __init__(self, 
                 n_estimators=100,
                 learning_rate=0.001,
                 max_depth=2):
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.weak_learner = DecisionTreeRegressor(max_depth=self.max_depth)
        
        self.initial_value = None
        self.estimators = []
        self.terminal_node_estimates = []

    def fit(self, X, y):
        self.initial_value = self._initial_value(y)
        working_prediction = np.repeat(self.initial_value, y.shape[0])
        working_response = self._gradient(y, working_prediction)

        for idx in range(self.n_estimators):

            tree = clone(self.weak_learner).fit(X, working_response)
            terminal_node_estimates = self._terminal_node_estimates(
                X, y, working_prediction, tree)
            prediction_update = self._prediction_update(
                X, tree, terminal_node_estimates)

            working_prediction += self.learning_rate * prediction_update
            working_response = self._gradient(y, working_prediction)

            self.estimators.append(tree)
            self.terminal_node_estimates.append(terminal_node_estimates)
        return self

    def predict(self, X):
        y_hat = np.repeat(self.initial_value, X.shape[0])
        estimator_data = zip(self.estimators, self.terminal_node_estimates)
        for tree, tn_estimates in estimator_data:
            y_hat += self.learning_rate * self._prediction_update(X, tree, tn_estimates)
        return np.exp(y_hat)
            
    def _initial_value(self, y):
        return np.log(np.mean(y))

    def _gradient(self, y, y_hat):
        return y - np.exp(y_hat)

    def _terminal_node_estimates(self, X, y, y_hat, tree):
        exp_y_hat = np.exp(y_hat)

        estimates = {}
        terminal_node_assignments = tree.apply(X)
        terminal_node_idxs = np.unique(terminal_node_assignments)
        for idx in terminal_node_idxs:
            in_terminal_node = (terminal_node_assignments == idx)
            y_in_node = np.sum(y[in_terminal_node])
            exp_y_hat_in_node = np.sum(exp_y_hat[in_terminal_node])
            estimates[idx] = np.log(y_in_node / exp_y_hat_in_node)
        return estimates

    def _prediction_update(self, X, tree, terminal_node_estimates):
        prediction_update = np.zeros(X.shape[0])
        terminal_node_assignments = tree.apply(X)
        for idx in terminal_node_estimates:
            in_terminal_node = (terminal_node_assignments == idx)
            prediction_update[in_terminal_node] = terminal_node_estimates[idx]
        return prediction_update
