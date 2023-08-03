# Import necessary libraries
import os
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set up the input directory and category labels
input_dir = './clf-data'
categories = ['empty', 'not_empty']

# Prepare data
data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)  # Read the image using skimage's imread function
        img = resize(img, (15, 15))  # Resize the image to a 15x15 size using skimage's resize function
        data.append(img.flatten())  # Flatten the image into a one-dimensional array and add it to the data list
        labels.append(category_idx)  # Add the category index (0 for 'empty', 1 for 'not_empty') to the labels list

data = np.asarray(data)  # Convert the data list to a NumPy array
labels = np.asarray(labels)  # Convert the labels list to a NumPy array

# Train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
# Split the data and labels into training and testing sets (x_train, x_test, y_train, y_test)
# with 20% of the data as the test set. The data is shuffled, and the class distribution is preserved.

# Train classifier (Support Vector Machine with RBF kernel)
classifier = SVC()
# Initialize a Support Vector Machine (SVM) classifier with a radial basis function (RBF) kernel.
# RBF kernel is used for non-linear data.

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
# Define a list of hyperparameters for the SVM classifier.
# 'gamma' and 'C' are hyperparameters that control the SVM's ability to fit the training data.

grid_search = GridSearchCV(classifier, parameters)
# Create a GridSearchCV object that performs an exhaustive search over the hyperparameter space
# to find the best combination of hyperparameters that maximizes the model's performance using cross-validation.

grid_search.fit(x_train, y_train)
# Fit the GridSearchCV object to the training data (x_train, y_train),
# which performs the grid search and cross-validation to find the best hyperparameters.

# Test performance
best_estimator = grid_search.best_estimator_
# Obtain the best estimator (model) from the GridSearchCV object, which is the trained SVM model
# with the best hyperparameters found during the grid search.

y_prediction = best_estimator.predict(x_test)
# Use the best estimator to predict the labels for the test data (x_test).

score = accuracy_score(y_prediction, y_test)
# Calculate the accuracy of the model by comparing the predicted labels (y_prediction)
# with the true labels from the test set (y_test).

print('{}% of samples were correctly classified'.format(str(score * 100)))
# Print the accuracy score as a percentage of correctly classified samples in the test set.

pickle.dump(best_estimator, open('./model.p', 'wb'))
# Save the best estimator (trained SVM model) to a file named 'model.p' using pickle.
# This allows the model to be reused later without the need for retraining.
# The 'wb' mode in open() indicates that the file is opened for writing in binary mode.