import numpy as np
import cv2
import os
import json
from generate_coords import get_cords


def generate_mean(name, data_dir):
    coordinates_list = get_cords()
    image = cv2.cvtColor(cv2.imread(os.path.join(data_dir, name)), cv2.COLOR_BGR2GRAY)

    mean = {}
    i = 1
    for coordinates in coordinates_list:
        mean[f"F_{i}"] = np.mean(image[coordinates[:, 0], coordinates[:, 1]])
        i += 1

    return mean

def compare_means(mean, data_dir):

    comparison_results = {}
    for json_file in os.listdir(data_dir):
        if json_file.endswith('.json'):
            with open(os.path.join(data_dir, json_file), 'r') as file:
                json_data = json.load(file)
                for key in mean:
                    if key in json_data:
                        comparison_results[f"{json_file}_{key}"] = abs(mean[key] - json_data[key])
                    else:
                        comparison_results[f"{json_file}_{key}"] = None  # or some other default value indicating no match
   
    return comparison_results

def find_face(comparison_results):
    faces = {}
    for key, value in comparison_results.items():
        subject, feature = key.split('_F_')
        if subject not in faces:
            faces[subject] = []
        faces[subject].append(value)
    
    for face, differences in faces.items():
        if all(diff is not None and diff < 3 for diff in differences):
            return face
    return None

if __name__ == '__main__':
    mean = generate_mean(name="subject06.png", data_dir=r"data")
    comparison_results = compare_means(mean, data_dir=r"json_data")
    print(find_face(comparison_results))
