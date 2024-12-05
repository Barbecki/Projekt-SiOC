import numpy as np
import json


# funkcja generująca losowe koordynaty
def generate_random_coordinates(h, w, n):
    _rand = np.random.default_rng().choice(w*h, size=n, replace=False)
    y_random = _rand // w
    x_random = _rand % w
    return y_random, x_random

# funkcja zapisująca liste koordynatów do pliku json
def create_cord_list():
    num_samples = 100
    num_means = 100
    dims = (100, 100)

    # generowanie listy koordynatów
    coordinates_list = [generate_random_coordinates(*dims, num_samples) for _ in range(num_means)]

    # przygotowanie listy do zapisu w formacie json
    coordinates_json = {}
    i = 1
    for coordinates in coordinates_list:
        coordinates_json[f"X_{i}"] = coordinates[0].tolist()
        coordinates_json[f"Y_{i}"] = coordinates[1].tolist()
        i += 1

    # zapisywanie do pliku json
    with open("coordinates.json", "w") as f:
        f.write(json.dumps(coordinates_json, indent=4))
    return coordinates_list

# funkcja pobierająca liste koordynatów z pliku json
def get_cords(file_path='coordinates.json'):
    with open(file_path, 'r') as file:
        coordinates = json.load(file)
    coordinates_list = [np.array(list(zip(coordinates[f'X_{i}'], coordinates[f'Y_{i}']))) for i in range(1, (len(coordinates) // 2) + 1)]
    return coordinates_list