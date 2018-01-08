import numpy as np


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

    def fit(X, y):
        self.initial_value = self._initial_value(y)
        working_prediction = np.repeat(self.initial_value, y.shape[0])
        working_response = y - np.exp(working_prediction)

        for _ in range(self.n_estimators):
            pass


    def predict(X):
        pass

    def _initial_value(self, y):
        pass

    def _gradient(self, y, y_hat):
        pass

    def _get_terminal_nodes(self, X, tree):
        pass

    def _terminal_node_estimates(self, y, y_hat, terminal_node_mask):
        pass
