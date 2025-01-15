import os
import numpy as np
import cv2
import csv
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import scipy.stats as stats

def generate_random_coordinates(h: int, w: int, n: int) -> tuple:
    """
    Generating tuple of n random coordinates
    :param h: height of the matrix
    :param w: width of the matrix
    :param n: number of random coordinates to generate
    :return: (ndarray, ndarray); generated coordinates (row index, column index)
    """

    h = h // 2
    w = w // 2
    _rand = np.random.default_rng().choice(w * h, size=n, replace=False)
    y_random = _rand // w
    x_random = _rand % w
    return y_random, x_random

def generate_means(data_dir, coordinates_list) -> None:
    """
    Generating means of values from data from coordinates and saving them in json file
    :param data_dir: path to directory with data (grayscale images with .png extension)
    :param coordinates_list: list of lists of coordinates to calculate means from
    :return: None
    """
    with open('data.csv', 'w', newline='') as f:
        for name in os.listdir(data_dir):
            if name.endswith('.gif'):
                ret, image = cv2.VideoCapture(os.path.join(data_dir, name)).read()
                if not ret:
                    raise AssertionError("Error with loading image.")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Discrete Cosine Transform
                dct = cv2.dct(np.float32(image))
                h, w = dct.shape[:2]
                image = dct[:h // 2, :w // 2]

                means = []
                for coordinates in coordinates_list:
                    means.append(float(np.mean(image[coordinates])))
                csv.writer(f, delimiter=';').writerow(means)

def correlation_coefficient(first: np.ndarray, second: np.ndarray, m: int) -> float:
    """
    Calculate correlation coefficient for a pair of arrays.
    :param first: first list of means
    :param second: second list of means
    :param m: number of means
    :return: calculated correlation coefficient
    """
    # Normalisation; mean and standard deviation of normalised data is respectively 0 and 1
    # (in our case it is approximately 0 and 1)
    mean_f = np.mean(first)
    std_f = np.std(first)
    if std_f:
        normalised_f = (first - mean_f)/std_f
    else:
        normalised_f = np.zeros_like(first)
    mean_s = np.mean(second)
    std_s = np.std(second)
    if std_s:
        normalised_s = (second - mean_s)/std_s
    else:
        normalised_s = np.zeros_like(second)

    return float(sum([i*j for i, j in zip(normalised_f, normalised_s)])/m)


if __name__ == '__main__':
    DATA_DIR = 'data'
    NUMBER_OF_SAMPLES = 100
    NUMBER_OF_MEANS = 50
    CRDS = [generate_random_coordinates(*(243, 320), NUMBER_OF_SAMPLES) for _ in range(NUMBER_OF_MEANS)]
    generate_means(DATA_DIR, CRDS)
    data = np.loadtxt('data.csv', delimiter=';')
    labels = [name for name in os.listdir(DATA_DIR) if name.endswith('.gif')]

    # for each pair of images we create a list with ['Pair Name', 'Similarity Score (rho)', 'Ground Truth Match']
    pairs = [[f"{img1}-{img2}",
              correlation_coefficient(data[labels.index(img1)], data[labels.index(img2)], NUMBER_OF_MEANS),
              int(img1[:9] == img2[:9])]
             for img1 in labels for img2 in labels[labels.index(img1):] if img1 != img2]

    with open('pairs_comparison.csv', 'w', newline='') as file:
        csv.writer(file, delimiter=';').writerows(pairs)

    # Plotting Receiver Operating Characteristic (ROC) Curve and calculating Area Under Curve (AUC)
    # False Positive Rate (FPR) = False Positives/(False Positives + True Negatives) = Incorrect Matches/Total Non-Matches
    # True Positive Rate (TPR)  = True Positives/(True Positives + False Negatives) = Correct Matches/Total Matches
    # AUC = 1 - perfect classifier; AUC = 0.5 - random classifier (randomly guessing is as good as this); AUC < 0.5 - model is worse than random

    fpr, tpr, thresholds = roc_curve([pair[2] for pair in pairs], [pair[1] for pair in pairs])
    auc = roc_auc_score([pair[2] for pair in pairs], [pair[1] for pair in pairs])

    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], 'r--', label="Random Guessing")
    plt.scatter(x=0.05, y=tpr[np.argmin(np.abs(fpr - 0.05))], color='green',
                label=f"Point (0.05, {tpr[np.argmin(np.abs(fpr - 0.05))]:.2f})\nThreshold = {thresholds[np.argmin(np.abs(fpr - 0.05))]:.3f}")
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve for Face Recognition')
    plt.legend()
    plt.savefig('ROC_curve.png')
    plt.close()

    same = [pair[1] for pair in pairs if pair[2] == 1]
    different = [pair[1] for pair in pairs if pair[2] == 0]

    t_stat, p_value = stats.mannwhitneyu(same, different, alternative='greater', method='asymptotic')

    print(f"Mann-Whitney Test; z: {t_stat}, p-value: {p_value}")

    if p_value < 0.05:
        print("Rejecting H0 - there is significant difference between the groups. -> H1")
    else:
        print("No reason to reject H0 - no significant difference between the groups. -> H0")

    s1, s2 = stats.t.interval(0.95, df=len(same) - 1, loc=np.mean(same), scale=np.std(same, ddof=1) / np.sqrt(len(same)))
    d1, d2 = stats.t.interval(0.95, df=len(different) - 1, loc=np.mean(different), scale=np.std(different, ddof=1) / np.sqrt(len(different)))
    print(f"\nConfidence intervals:\nsame-person-pairs: {(float(s1), float(s2))}, mean: {np.mean(same)}\n"
          f"different-person-pairs: {(float(d1), float(d2))}, mean: {np.mean(different)}")

    plt.figure(figsize=(9,8))
    plt.hist(same, bins=20, alpha=0.5, color='blue', label='Same-person pairs')
    plt.hist(different, bins=20, alpha=0.5, color='red', label='Different-person pairs')
    plt.xticks([i/10 for i in range(-10, 11)])
    plt.yticks([i for i in range(0, 2000, 100)])
    plt.grid(True)

    plt.xlabel('Similarity Score (Correlation Coefficient)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Similarity Scores - frequency')
    plt.legend(loc='upper left')
    plt.savefig('histograms.png')
    plt.close()

    plt.figure(figsize=(9,8))

    same_counts, same_bins = np.histogram(same, bins=20)
    same_heights = same_counts / same_counts.sum() * 100
    diff_counts, diff_bins = np.histogram(different, bins=20)
    diff_heights = diff_counts / diff_counts.sum() * 100
    plt.bar(same_bins[:-1], same_heights, width=np.diff(same_bins), alpha=0.5, color='blue', label='Same-person pairs', align='edge')
    plt.bar(diff_bins[:-1], diff_heights, width=np.diff(diff_bins), alpha=0.5, color='red', label='Different-person pairs', align='edge')
    plt.yticks([i for i in range(0, 31, 2)], [f"{i}%" for i in range(0, 31, 2)])
    plt.grid(True)

    plt.xlabel('Similarity Score (Correlation Coefficient)')
    plt.ylabel('Percentage')
    plt.title('Histogram of Similarity Scores - percentage')
    plt.legend(loc='upper left')
    plt.savefig('histograms_percentage.png')
    plt.close()