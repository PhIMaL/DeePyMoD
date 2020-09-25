'''Sparsity estimators which can be plugged into deepmod.
We keep the API in line with scikit learn (mostly), so scikit learn can also be plugged in.
See scikitlearn.linear_models for applicable estimators.'''

import numpy as np
from .deepmod import Estimator
from sklearn.cluster import KMeans
from pysindy.optimizers import STLSQ
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # To silence annoying pysindy warnings


class Base(Estimator):
    """[summary]

    Args:
        Estimator ([type]): [description]
    """
    def __init__(self, estimator: BaseEstimator) -> None:
        super().__init__()
        self.estimator = estimator
        self.estimator.set_params(fit_intercept=False)  # Library contains offset so turn off the intercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """[summary]

        Args:
            X (np.ndarray): [description]
            y (np.ndarray): [description]

        Returns:
            np.ndarray: [description]
        """
        coeffs = self.estimator.fit(X, y).coef_
        return coeffs


class Threshold(Estimator):
    '''Performs additional thresholding on coefficient result from estimator. Basically
    a thin wrapper around the given estimator. '''
    def __init__(self, threshold: float = 0.1, estimator: BaseEstimator = LassoCV(cv=5, fit_intercept=False)) -> None:
        super().__init__()
        self.estimator = estimator
        self.threshold = threshold

        # Library contains offset so turn off the intercept
        self.estimator.set_params(fit_intercept=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """[summary]

        Args:
            X (np.ndarray): [description]
            y (np.ndarray): [description]

        Returns:
            np.ndarray: [description]
        """
        coeffs = self.estimator.fit(X, y).coef_
        coeffs[np.abs(coeffs) < self.threshold] = 0.0

        return coeffs


class Clustering(Estimator):
    ''' Performs additional thresholding by clustering on coefficient result from estimator. Basically
    a thin wrapper around the given estimator. Results are fitted to two groups:
    components to keep and components to throw.
    '''
    def __init__(self, estimator: BaseEstimator = LassoCV(cv=5, fit_intercept=False)) -> None:
        super().__init__()
        self.estimator = estimator
        self.kmeans = KMeans(n_clusters=2)

        # Library contains offset so turn off the intercept
        self.estimator.set_params(fit_intercept=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """[summary]

         Args:
             X (np.ndarray): [description]
             y (np.ndarray): [description]

         Returns:
             np.ndarray: [description]
        """
        coeffs = self.estimator.fit(X, y).coef_[:, None]  # sklearn returns 1D
        clusters = self.kmeans.fit_predict(np.abs(coeffs)).astype(np.bool)

        # make sure terms to keep are 1 and to remove are 0
        max_idx = np.argmax(np.abs(coeffs))
        if clusters[max_idx] != 1:
            clusters = ~clusters

        coeffs = clusters.astype(np.float32)
        return coeffs


class PDEFIND():
    ''' Implements PDEFIND as a sparse estimator.'''
    def __init__(self, lam: float = 1e-3, dtol: float = 0.1) -> None:
        self.lam = lam
        self.dtol = dtol

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """[summary]

        Args:
            X (np.ndarray): [description]
            y (np.ndarray): [description]

        Returns:
            np.ndarray: [description]
        """
        coeffs = PDEFIND.TrainSTLSQ(X, y[:, None], self.lam, self.dtol)
        return coeffs.squeeze()

    @staticmethod
    def TrainSTLSQ(X: np.ndarray,
                   y: np.ndarray,
                   alpha: float,
                   delta_threshold: float,
                   max_iterations: int = 100,
                   test_size: float = 0.2,
                   random_state: int = 0) -> np.ndarray:
        """[summary]

        Args:
            X (np.ndarray): [description]
            y (np.ndarray): [description]
            alpha (float): [description]
            delta_threshold (float): [description]
            max_iterations (int, optional): [description]. Defaults to 100.
            test_size (float, optional): [description]. Defaults to 0.2.
            random_state (int, optional): [description]. Defaults to 0.

        Returns:
            np.ndarray: [description]
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Set up the initial tolerance l0 penalty and estimates
        l0 = 1e-3 * np.linalg.cond(X)
        delta_t = delta_threshold  # for interal use, can be updated

        # Initial estimate
        optimizer = STLSQ(threshold=0, alpha=0.0, fit_intercept=False)  # Now similar to LSTSQ
        y_predict = optimizer.fit(X_train, y_train).predict(X_test)
        min_loss = np.linalg.norm(y_predict - y_test, 2) + l0 * np.count_nonzero(optimizer.coef_)

        # Setting alpha and tolerance
        best_threshold = delta_t
        threshold = delta_t

        for iteration in np.arange(max_iterations):
            optimizer.set_params(alpha=alpha, threshold=threshold)
            y_predict = optimizer.fit(X_train, y_train).predict(X_test)
            loss = np.linalg.norm(y_predict - y_test, 2) + l0 * np.count_nonzero(optimizer.coef_)

            if (loss <= min_loss) and not (np.all(optimizer.coef_ == 0)):
                min_loss = loss
                best_threshold = threshold
                threshold += delta_threshold

            else:  # if loss increases, we need to a) lower the current threshold and/or decrease step size
                new_lower_threshold = np.max([0, threshold - 2 * delta_t])
                delta_t = 2 * delta_t / (max_iterations - iteration)
                threshold = new_lower_threshold + delta_t

        optimizer.set_params(alpha=alpha, threshold=best_threshold)
        optimizer.fit(X_train, y_train)

        return optimizer.coef_
