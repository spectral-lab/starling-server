import numpy as np
from typing import List, Callable
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from pdb import set_trace


def compute_feature_lines(training_points: List[np.ndarray], degree: int = 0) -> List[np.ndarray]:
    """
    :param training_points:
    :param degree:
    :return: [row, column] pairs splitted into some chunks which represents a peak line.
    """
    list_of_feature_lines = []
    for i in range(len(training_points)):
        target_training_points = training_points[i]
        if target_training_points.shape[0] < 1:
            raise Exception('No training points')
        x = target_training_points[:, 0]
        coefs_y = polynomial_regression(x, target_training_points[:, 1], degree)
        coefs_z = polynomial_regression(x, target_training_points[:, 2], degree)
        calc_y: Callable = coefs_to_formula(coefs_y)
        calc_z: Callable = coefs_to_formula(coefs_z)
        x_seq = np.sort(np.unique(x))
        list_of_feature_lines.append(np.column_stack([x_seq, calc_y(x_seq), calc_z(x_seq)]))
    return list_of_feature_lines


def polynomial_regression(x: np.ndarray, y: np.ndarray, degree=0) -> np.ndarray:
    model = Pipeline([('poly', PolynomialFeatures(degree)),
                      ('linear', LinearRegression(fit_intercept=False))])
    model = model.fit(x[:, np.newaxis], y)
    coefs = model.named_steps['linear'].coef_
    return coefs


def coefs_to_formula(coefs: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def ret_formula(x: np.ndarray) -> np.ndarray:
        y = np.zeros(x.shape)
        for i in range(coefs.shape[0]):
            y += coefs[i] * x ** i
        return y

    return ret_formula
