import numpy as np

def find_face(first: dict, second: dict, m: int, threshold: float) -> bool:
    """
    Compare two dictionaries of means and determine whether it's the same face or not.
    :param first: first dictionary of means
    :param second: second dictionary of means
    :param m: number of means
    :param threshold: threshold for correlation coefficient
    :return: boolean whether both of these dictionaries represent the same face
    """
    # Normalisation; mean and standard deviation of normalised data is respectively 0 and 1
    # (in our case it is approximately 0 and 1)
    first_v = list(first.values())
    mean_f = np.mean(first_v)
    std_f = np.std(first_v)
    if std_f:
        normalised_f = (first_v - mean_f)/std_f
    else:
        normalised_f = np.zeros_like(first_v)
    second_v = list(second.values())
    mean_s = np.mean(second_v)
    std_s = np.std(second_v)
    if std_s:
        normalised_s = (second_v - mean_s)/std_s
    else:
        normalised_s = np.zeros_like(second_v)


    # Calculating correlation coefficient;
    # rho = 1 perfect positive correlation, rho = 0 no correlation, rho = -1 perfect negative correlation (unlikely in face recognition)
    rho = sum([i*j for i, j in zip(normalised_f, normalised_s)])/m
    return rho >= threshold



