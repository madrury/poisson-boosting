import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingPoissonRegressor:
    """A Gradient Boosted Regression Tree model for conditionally Poisson
    distributed data.

    This model estimates a function f for which

        y | X ~ Poisson(lambda = exp(f(X)))

    The function f is estimated as a sum of regression trees by maximizing the
    log-likelihood of the poisson model.

    Parameters
    ----------
    n_estimators: int
        A positive integer.  The number of regression trees to stack when
        estimating f.

    learning_rate: float
        A non-negative number which is less than one.  The amount of shrinkage
        to apply to an estimated regression tree before adding into f.

    max_depth: int
        Positive integer.  The maximum depth for the estimated regression trees.

    Attributes
    ----------
    n_estimators: int
        The number of regression trees to stack when estimating f.

    learning_rate: float
        The amount of shrinkage to apply to an estimated regression tree before
        adding into f.

    max_depth: int
        The maximum depth for the estimated regression trees.

    initial_value_: float
        The overall mean of y in the training data.  Used as an initial
        prediction, all subsequent regression trees predictions are added to
        this baseline value.

    estimators_: list of sklean.tree objects.
        Sequence of estimated regression trees.

    terminal_node_estimates_: list of dict(int: float)
        Predictions that apply in the terminal nodes (leaves) of the estimated
        regression trees.  These prediction are summed into the final
        predictions of the gradient boosted model.
    """
    def __init__(self, 
                 n_estimators=100,
                 learning_rate=0.001,
                 max_depth=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.weak_learner = DecisionTreeRegressor(max_depth=self.max_depth)
        # Estimated quantities.
        self.initial_value_ = None
        self.estimators_ = []
        self.terminal_node_estimates_ = []

    def fit(self, X, y):
        """Fit boosted model to training data.

        Parameters
        ----------
        X: array, shape (n_samples, n_features)
            Training data.

        y: array, shape (n_samples, )
            Target values.
        """
        self.initial_value_ = self._initial_value(y)
        working_prediction = np.repeat(self.initial_value_, y.shape[0])
        working_response = self._gradient(y, working_prediction)
        for idx in range(self.n_estimators):
            tree = clone(self.weak_learner).fit(X, working_response)
            terminal_node_estimates = self._terminal_node_estimates(
                X, y, working_prediction, tree)
            prediction_update = self._prediction_update(
                X, tree, terminal_node_estimates)
            working_prediction += self.learning_rate * prediction_update
            working_response = self._gradient(y, working_prediction)
            self.estimators_.append(tree)
            self.terminal_node_estimates_.append(terminal_node_estimates)
        return self

    def predict(self, X):
        """Return predictions from a fit model.

        Parameters
        ----------
        X: array, shape (n_samples, n_features)
            Data set.
        """
        log_prediction = np.repeat(self.initial_value_, X.shape[0])
        estimator_data = zip(self.estimators_, self.terminal_node_estimates_)
        for tree, tn_estimates in estimator_data:
            log_prediction += (
                self.learning_rate * 
                self._prediction_update(X, tree, tn_estimates))
        return np.exp(log_prediction)
            
    def _initial_value(self, y):
        return np.log(np.mean(y))

    def _gradient(self, y, log_prediction):
        return y - np.exp(log_prediction)

    def _terminal_node_estimates(self, X, y, log_prediction, tree):
        prediction = np.exp(log_prediction)
        estimates = {}
        terminal_node_assignments = tree.apply(X)
        terminal_node_idxs = np.unique(terminal_node_assignments)
        for idx in terminal_node_idxs:
            in_terminal_node = (terminal_node_assignments == idx)
            y_in_node = np.sum(y[in_terminal_node])
            prediction_in_node = np.sum(prediction[in_terminal_node])
            estimates[idx] = np.log(y_in_node / prediction_in_node)
        return estimates

    def _prediction_update(self, X, tree, terminal_node_estimates):
        prediction_update = np.zeros(X.shape[0])
        terminal_node_assignments = tree.apply(X)
        for idx in terminal_node_estimates:
            in_terminal_node = (terminal_node_assignments == idx)
            prediction_update[in_terminal_node] = terminal_node_estimates[idx]
        return prediction_update
