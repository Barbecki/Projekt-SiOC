import numpy as np
import cv2
import os
import json

def generate_random_coordinates(h, w, n):
    _rand = np.random.default_rng().choice(w*h, size=n, replace=False)
    y_random = _rand // w
    x_random = _rand % w
    return y_random, x_random

if __name__ == '__main__':
    DIR = r"data"
    OUTPUT_DIR = r"json_data"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    num_samples = 100
    num_means = 100
    height, width = 100, 100
    coordinates_list = [generate_random_coordinates(height, width, num_samples) for i in range(num_means)]
    for name in os.listdir(DIR):
        image = cv2.cvtColor(cv2.imread(os.path.join(DIR, name)), cv2.COLOR_BGR2GRAY)
        means = {}
        i = 1
        for coordinates in coordinates_list:
            means[f"F_{i}"] = np.mean(image[coordinates])
            i += 1
        with open(os.path.join(OUTPUT_DIR, f"{name[:-4]}.json"), "w") as f:
            f.write(json.dumps(means, indent=4))
