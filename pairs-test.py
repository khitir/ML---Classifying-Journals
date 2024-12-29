# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:03:09 2024

@author: Zakaria
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Function to load data from two specific categories
def load_local_20newsgroups(data_dir, selected_categories):
    data = []
    target = []
    target_names = selected_categories
    for idx, category in enumerate(selected_categories):
        category_dir = os.path.join(data_dir, category)
        if not os.path.isdir(category_dir):
            continue
        files = glob.glob(os.path.join(category_dir, '*'))
        if not files:
            print(f"No files found in category directory: {category_dir}")
        for filepath in files:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                data.append(file.read())
                target.append(idx)
    return data, target, target_names

# Path to your local 20 newsgroups data directory
data_dir = r'C:\Users\Zakaria\Desktop\20_newsgroups'

# Specify the two categories you want to use
selected_categories = ['sci.space', 'comp.graphics']

# Load the data
data, target, target_names = load_local_20newsgroups(data_dir, selected_categories)

# Check if data was loaded correctly
if not data:
    raise ValueError(f"No data loaded. Please check the data directory: {data_dir}")
if not target:
    raise ValueError(f"No target labels found. Please check the data directory: {data_dir}")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Create a pipeline with TfidfVectorizer and MultinomialNB
model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred, target_names=target_names))

# Plot confusion matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

# Function to predict the category of a given text
def predict_category(s, model=model, target_names=target_names):
    pred = model.predict([s])
    return target_names[pred[0]]

# Example usage of predict_category function
print(predict_category('air cosmos space stars send'))
print(predict_category('plane, triange, graph, plot, axis'))
