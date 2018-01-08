import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingPoissonRegressor:


    def __init__(self, 
                 n_estimators=100,
                 learing_rate=0.01,
                 tree_depth=2,
                 subsample=0.5):
        
        self.n_estimators = n_estimators
        self.learing_rate = learing_rate
        self.tree_depth = tree_depth
        self.subsample = subsample
        self.weak_learner = DecisionTreeRegressor(max_depth=self.max_depth)
        
        self.initial_value = None
        self.estimators = []
        self.terminal_node_estimates = []

    def fit(X, y):
        self.initial_value = self._initial_value(y)
        working_prediction = np.repeat(self.initial_value, y.shape[0])
        working_response = self._gradient(y, working_prediction)

        for _ in range(self.n_estimators):
            tree = clone(self.weak_learner).fit(X, working_response)
            terminal_node_estimates = self._terminal_node_estimates(
                y, working_prediction, terminal_node_mask)
            prediction_update = self._prediction_update(
                X, tree, terminal_node_estimates)
            working_prediction += terminal_node_estimates
            working_response = self._gradient(y, working_prediction)

            self.estimators.append(tree)
            self.terminal_node_estimates[idx, :] = terminal_node_estimates

    def predict(self, X):
        y_hat = np.repeat(self.initial_value)
        for tree, tn_estimates in zip(self.estimators, terminal_node_estimates):
            y_hat += self._prediction_update(X, tree, tn_estimates)
        return np.exp(y_hat)
            
    def _initial_value(self, y):
        return np.log(np.mean(y))

    def _gradient(self, y, y_hat):
        return y - np.exp(y_hat)

    def _terminal_node_estimates(self, X, y, y_hat, tree):
        exp_y_hat = np.exp(y_hat)

        estiamtes = {}
        terminal_node_assignments = tree.apply(X)
        terminal_node_idxs = np.unique(terminal_node_assignments)
        for idx in terminal_node_idxs:
            in_terminal_node = (terminal_node_assignments == idx)
            y_in_node = np.sum(y[in_terminal_node])
            exp_y_hat_in_node = np.sum(exp_y_hat[in_terminal_node])
            estiamtes[idx] = np.log(y_in_node / exp_y_hat_in_node)
        return estiamtes

    def _prediction_update(self, X, tree, terminal_node_estimates):
        terminal_node_assignments = tree.apply(X)
        for idx in terminal_node_estimates:
            in_terminal_node = (terminal_node_assignments == idx)
            prediction_update[in_terminal_node] = terminal_node_estimates[idx]
        return prediction_update
