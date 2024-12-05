import os
import json
from generate_means import create_coordinate_list, generate_means
from face_recognition import find_face

if __name__ == '__main__':
    DATA_DIR = r"data"
    JSON_DIR = r"json_data"
    OG_IMG = r"subject01.png"
    DIMENSIONS = (100, 100)
    NUMBER_OF_SAMPLES = 100
    NUMBER_OF_MEANS = 100
    THRESHOLD = 0.9 # should be in range [0.7, 1)
    CRDS = create_coordinate_list(DIMENSIONS, NUMBER_OF_SAMPLES, NUMBER_OF_MEANS)

    generate_means(DATA_DIR, JSON_DIR, CRDS)
    face_to_compare = json.load(open(os.path.join(JSON_DIR, f"{os.path.splitext(OG_IMG)[0]}.json"), "r"))
    for name in os.listdir(DATA_DIR):
        if name.endswith(".png"):
            other_face = json.load(open(os.path.join(JSON_DIR, f"{os.path.splitext(name)[0]}.json"), "r"))
            print(f"Comparing {OG_IMG} and {name}\nmatch = {find_face(face_to_compare, other_face, NUMBER_OF_MEANS, THRESHOLD)}")
    # TODO: Upload more data from Kaggle https://www.kaggle.com/datasets/olgabelitskaya/yale-face-database
    # TODO: Test for different NUMBER_OF_SAMPLES, NUMBER_OF_MEANS, THRESHOLD
