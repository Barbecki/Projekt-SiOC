import numpy as np
import cv2
import os
import json


def generate_random_coordinates(h: int, w: int, n: int) -> tuple:
    """
    Generating tuple of n random coordinates
    :param h: height of the matrix
    :param w: width of the matrix
    :param n: number of random coordinates to generate
    :return: (ndarray, ndarray); generated coordinates (row index, column index)
    """
    _rand = np.random.default_rng().choice(w * h, size=n, replace=False)
    y_random = _rand // w
    x_random = _rand % w
    return y_random, x_random

def create_coordinate_list(dims: tuple, num_samples: int, m: int) -> list:
    """
    Generating list of arrays containing random coordinates
    :param dims: dimensions of the matrix (height, width)
    :param num_samples: number of coordinates to randomly choose for each array
    :param m: number of arrays
    :return: List[(ndarray, ndarray)]; m arrays, each with n number of coordinates [(column indexes, row indexes)]
    """
    coordinates_list = [generate_random_coordinates(*dims, num_samples) for _ in range(m)]

    return coordinates_list

def generate_means(data_dir, output_dir, coordinates_list) -> None:
    """
    Generating means of values from data from coordinates and saving them in json file
    :param data_dir: path to directory with data (grayscale images with .png extension)
    :param output_dir: path to directory to save the json files to - if it doesn't exist it will be created
    :param coordinates_list: list of coordinates to calculate mean from
    :return: None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for name in os.listdir(data_dir):
        if name.endswith(".png"):
            image = cv2.cvtColor(cv2.imread(os.path.join(data_dir, name)), cv2.COLOR_BGR2GRAY)

            means = {}
            i = 1
            for coordinates in coordinates_list:
                means[f"F_{i}"] = np.mean(image[coordinates])
                i += 1

            with open(os.path.join(output_dir, f"{os.path.splitext(name)[0]}.json"), "w") as f:
                f.write(json.dumps(means, indent=4))
