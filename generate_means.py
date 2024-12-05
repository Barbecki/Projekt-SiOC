import numpy as np
import cv2
import os
import json

def generate_random_coordinates(h, w, n):
    _rand = np.random.default_rng().choice(w*h, size=n, replace=False)
    y_random = _rand // w
    x_random = _rand % w
    return y_random, x_random

def generate_means(data_dir, output_dir, num_samples, num_means, dims):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    coordinates_list = [generate_random_coordinates(*dims, num_samples) for _ in range(num_means)]
    for name in os.listdir(data_dir):
        image = cv2.cvtColor(cv2.imread(os.path.join(data_dir, name)), cv2.COLOR_BGR2GRAY)
        means = {}

        i = 1
        for coordinates in coordinates_list:
            means[f"F_{i}"] = np.mean(image[coordinates])
            i += 1

        with open(os.path.join(output_dir, f"{name[:-4]}.json"), "w") as f:
            f.write(json.dumps(means, indent=4))

if __name__ == '__main__':
    generate_means(data_dir=r"data", output_dir=r"json_data", num_samples=100, num_means=100, dims=(100, 100))
