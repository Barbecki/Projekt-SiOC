import os
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.feature import hog
from skimage import exposure
from sklearn.metrics import roc_curve, auc


def save_magnitude_spectrum(image_path, output_path):
    """
    Save the magnitude spectrum of a single image to a file
    :param image_path: path to the image file
    :param output_path: path to save the magnitude spectrum image
    :return: None
    """
    ret, image = cv2.VideoCapture(image_path).read()
    if not ret:
        raise AssertionError("Error with loading image.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Fourier Transform
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))
    
    # Normalize the magnitude spectrum to the range [0, 255]
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    
    # Save the magnitude spectrum as an image
    cv2.imwrite(output_path, magnitude_spectrum)

def compare_magnitude_spectrums(image_path1, image_path2):
    """
    Compare the magnitude spectrums of two images
    :param image_path1: path to the first image file
    :param image_path2: path to the second image file
    :return: similarity score
    """
    def get_magnitude_spectrum(image_path):
        ret, image = cv2.VideoCapture(image_path).read()
        if not ret:
            raise AssertionError("Error with loading image.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f_transform = np.fft.fft2(image)
        f_transform_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        return magnitude_spectrum

    magnitude_spectrum1 = get_magnitude_spectrum(image_path1)
    magnitude_spectrum2 = get_magnitude_spectrum(image_path2)
    
    # Calculate similarity (e.g., using correlation)
    similarity = np.corrcoef(magnitude_spectrum1.flatten(), magnitude_spectrum2.flatten())[0, 1]
    return similarity



def compare_images_hog(image_path1, image_path2):
    """
    Compare two images using HOG
    :param image_path1: path to the first image file
    :param image_path2: path to the second image file
    :return: similarity score
    """
    def get_hog_features(image_path):
        ret, image = cv2.VideoCapture(image_path).read()
        if not ret:
            raise AssertionError("Error with loading image.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True)
        return fd

    hog_features1 = get_hog_features(image_path1)
    hog_features2 = get_hog_features(image_path2)
    
    # Calculate similarity using correlation
    similarity = np.corrcoef(hog_features1, hog_features2)[0, 1]
    return similarity

def plot_roc_curve(similarity_scores_file, output_path):
    """
    Plot ROC curve based on similarity scores
    :param similarity_scores_file: path to the CSV file with similarity scores
    :param output_path: path to save the ROC curve plot
    :return: None
    """
    # Read similarity scores from CSV
    image_pairs = []
    similarities = []
    with open(similarity_scores_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            image_pairs.append((row[0], row[1]))
            similarities.append(float(row[2]))

    # Generate labels (1 for same subject, 0 for different subjects)
    labels = []
    for img1, img2 in image_pairs:
        if img1.split('_')[0] == img2.split('_')[0]:
            labels.append(1)
        else:
            labels.append(0)

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    DATA_DIR = 'data'
    OUTPUT_DIR = 'magnitude_spec'
    labels = [name for name in os.listdir(DATA_DIR) if name.endswith('.gif')]
    image_shape = (243, 320)
    
    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    for label in labels:
        image_path = os.path.join(DATA_DIR, label)
        output_path = os.path.join(OUTPUT_DIR, f'magnitude_spectrum_{label}.png')
        save_magnitude_spectrum(image_path, output_path)
    
    # Compare each image with every other image and save results to CSV using HOG
    similarity_scores_file = 'similarity_scores_hog.csv'
    #with open(similarity_scores_file, 'w', newline='') as csvfile:
    #    csv_writer = csv.writer(csvfile)
    #    csv_writer.writerow(['Image1', 'Image2', 'Similarity'])
    #    
    #    for i in range(len(labels)):
    #        for j in range(i + 1, len(labels)):
    #            image_path1 = os.path.join(DATA_DIR, labels[i])
    #            image_path2 = os.path.join(DATA_DIR, labels[j])
    #            similarity_score = compare_images_hog(image_path1, image_path2)
    #            csv_writer.writerow([labels[i], labels[j], similarity_score])
    
    # Plot ROC curve
    plot_roc_curve(similarity_scores_file, 'roc_curve_hog.png')


