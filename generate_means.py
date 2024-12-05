import numpy as np
import cv2
import os
import json
from generate_coords import get_cords,create_cord_list

def generate_means(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    coordinates_list = get_cords()
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
    generate_means(data_dir=r"data", output_dir=r"json_data")