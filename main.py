import os
import json
from generate_means import create_coordinate_list, generate_means
from face_recognition import find_face

def main(source_directory, output_directory, image_to_test, dimensions, number_of_samples, number_of_means, test_criteria):
    generate_means(source_directory, output_directory, create_coordinate_list(dimensions, number_of_samples, number_of_means))
    face_to_compare = json.load(open(os.path.join(output_directory, f"{os.path.splitext(image_to_test)[0]}.json"), "r"))
    tp, tn, fp, fn = 0, 0, 0, 0

    for name in os.listdir(source_directory):
        if name.endswith(".gif"):
            other_face = json.load(open(os.path.join(output_directory, f"{os.path.splitext(name)[0]}.json"), "r"))
            match = find_face(face_to_compare, other_face, number_of_means, test_criteria)

            # Incrementing counters
            if image_to_test.split('_')[0] == name.split('_')[0]:
                if match:
                    info = "True positive"
                    tp += 1
                else:
                    info = "False negative"
                    fn += 1
            else:
                if match:
                    info = "False positive"
                    fp += 1
                else:
                    info = "True negative"
                    tn += 1
            print(f"Comparing {image_to_test} and {name}\nMatch = {bool(match)}\ninfo: {info}")
    print(f"\nTrue positive's count: {tp}\nTrue negative's count: {tn}\nFalse positive's count: {fp}\nFalse negative's count: {fn}")

if __name__ == '__main__':
    main(r"data", r"json_data", r"subject01_centerlight.gif", (243, 320),
         100, 100, 0.8)
    # TODO: Test for different number_of_samples, number_of_means, test_criteria
