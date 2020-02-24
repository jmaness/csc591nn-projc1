from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split


class Main:
    """
    The main class for processing our data
    """

    def __init__(self):
        """
        Define hyperparameters for the class --- whether they be model hyperparameters, paths, or otherwise
        """
        self.train_dir = "TrainData/"
        self.train_ann = "TrainAnnotations.csv"

    def load_data(self):
        """
        Load the image training data and classes
        :return: list of numpy arrays, list of integer labels, list of string filenames (for reference)
        """
        file_labels = pd.read_csv(self.train_ann)
        ann_dict = pd.Series(file_labels.annotation.values, index=file_labels.file_name).to_dict()
        image_dict = dict.fromkeys(ann_dict.keys())
        print("Loading image data into dictionary...")
        for filename in list(ann_dict.keys()):
            image_dict[filename] = np.array(Image.open(self.train_dir + filename))
        print("Loading complete.")
        print("Preparing training data...")
        data = []
        labels = []
        names = []
        for filename in list(ann_dict):
            data.append(image_dict[filename])
            labels.append(ann_dict[filename])
            names.append(filename)
        print("Preparation complete.")
        return data, labels, names

    def visualize_data(self, array):
        """
        Plot a given numpy array
        :param array: a numpy array (image)
        """
        if len(array.shape) == 3:  # has all 3 RGB channels
            plt.imshow(array)
        else:  # just a black and white image (one channel)
            plt.imshow(array, cmap='gray')
        plt.axis("off")
        plt.show()

    def edge_filter(self, array, f_type="canny"):
        """
        Detect the edges within an image using a chosen filter
        :param array: a numpy array (image)
        :param f_type: the filter type in ['canny', 'laplacian', 'sobelx', 'sobely']
        :return: a numpy array (edge image)
        """
        edges = None
        if f_type == 'canny':
            edges = cv2.Canny(array, 120, 200)
        elif f_type == 'laplacian':
            # convert to grayscale --- one channel
            # Sobel filter in both directions
            array = np.array(Image.fromarray(array).convert('L'))
            edges = np.abs(cv2.Laplacian(array, cv2.CV_64F, ksize=5))
        elif f_type == "sobelx":
            # convert to grayscale --- one channel
            # get vertical edges
            array = np.array(Image.fromarray(array).convert('L'))
            edges = np.abs(cv2.Sobel(array, cv2.CV_64F, 1, 0, ksize=5))
        elif f_type == "sobely":
            # convert to grayscale --- one channel
            # get horizontal edges
            array = np.array(Image.fromarray(array).convert('L'))
            edges = np.abs(cv2.Sobel(array, cv2.CV_64F, 0, 1, ksize=5))
        return edges

    def extract_channel(self, array, channel='green'):
        """
        Extract one channel from the image (default green because that is arguably the most important
        for this application)
        :param array: a numpy array (image)
        :param channel: string in ('red', 'green', 'blue')
        :return: a numpy array (image, one channel)
        """
        channel_dict = {'red': 0, 'green': 1, 'blue': 2}
        matrix = array[:, :, channel_dict[channel]]
        return matrix

    def extract_sift(self, array):
        """
        Get the SURF features from an image
        :param array: a numpy array (image)
        :return: key points, image of the key points
        """
        sift = cv2.xfeatures2d.SIFT_create(1000)
        key_points = sift.detect(array, None)
        image = cv2.drawKeypoints(array, key_points, np.array([]), (0, 0, 255),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return key_points, image

    def extract_hog(self, array, visualize=False):
        """
        Get the Histogram of Gradients of the image
        :param array: a numpy array (image)
        :param visualize: to visualize the HOG image or not [True, False]
        :return: HOG vector (for training), HOG image (for visualization)
        """
        if visualize:
            vector, im = hog(array,
                             orientations=8,
                             pixels_per_cell=(4, 4),
                             cells_per_block=(1, 1),
                             block_norm='L2-Hys',
                             feature_vector=True,
                             visualize=True)
            return vector, im
        else:
            vector = hog(array,
                         orientations=8,
                         pixels_per_cell=(4, 4),
                         cells_per_block=(1, 1),
                         feature_vector=True,
                         block_norm='L2-Hys')
            return vector


if __name__ == "__main__":
    main = Main()
    X, y, filenames = main.load_data()

    # preview random datapoint
    # index = random.randint(0, len(X))
    # print("Previewing image", filenames[index], "from class", y[index])
    # main.visualize_data(X[index])

    # Edge feature extraction and visualization
    # edge_image = main.edge_filter(X[index], f_type='canny')
    # main.visualize_data(edge_image)

    # Extract channel and visualization
    # green_channel = main.extract_channel(X[index], channel='green')
    # main.visualize_data(green_channel)

    # Extract SIFT features of image and visualization
    # points, point_image = main.extract_sift(X[index])
    # print("SIFT Key Points:", points)
    # main.visualize_data(point_image)

    # Extract HOG features of image and visualization
    # vector, image = main.extract_hog(X[index])
    # print("HOG Feature Descriptor:", vector)
    # main.visualize_data(image)

    # Prepare data with concatenation of features of your choice
    # These are flattened to vectors
    print("Extracting features...")
    for i in range(len(X)):
        if i % 10 == 0:
            print("Preprocessing Image", i+1, '/', len(X))
        hog_vector = main.extract_hog(X[i], visualize=False)
        green = main.extract_channel(X[i], channel="green")
        X[i] = np.concatenate((hog_vector.flatten(), green.flatten()), axis=0)
    print("Vector size:", X[0].shape)

    # Split the data
    print("Splitting the data...")
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          train_size=0.8,
                                                          random_state=138,
                                                          shuffle=True,
                                                          stratify=y)

    # Train the model on the prepared data
    print("Training the model...")
    clf = svm.SVC(kernel="rbf")
    clf.fit(X_train, y_train)

    # Predict
    print("Validating the model...")
    y_hat = clf.predict(X_valid)
    accuracy = sum(y_valid == y_hat) / len(y_valid)
    print("Accuracy:", accuracy)
