# Naive Bayesian learning for classifying Netnews text articles
# Zakaria Khitirishvili
# CSCI 575 - Online

import os
import glob
import itertools
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

# Path to  local 20 newsgroups data directory
data_dir = r'C:\Users\Zakaria\Desktop\20_newsgroups'

# Get all category names
all_categories = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]

# Generate all possible pairs of categories
category_pairs = list(itertools.combinations(all_categories, 3))

# Function to train and evaluate model for a given pair of categories
def train_and_evaluate_for_pair(pair):
    data, target, target_names = load_local_20newsgroups(data_dir, pair)
    
    if not data:
        raise ValueError(f"No data loaded for categories: {pair}. Please check the data directory: {data_dir}")
    if not target:
        raise ValueError(f"No target labels found for categories: {pair}. Please check the data directory: {data_dir}")

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Results for categories: {pair}")
    print(classification_report(y_test, y_pred, target_names=target_names))

    mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.title(f'Confusion Matrix for {pair}')
    plt.show()
  

# Train and evaluate models for all pairs of categories
for pair in category_pairs[:5]:
    train_and_evaluate_for_pair(pair)
