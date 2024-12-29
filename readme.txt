1. Student: 				Zakaria Khitirishvili
   CWID: 				10862854

2. Programming Language: 		Python

3. The process begins by importing necessary libraries, such as os and glob for file handling, numpy for numerical operations, matplotlib and seaborn for plotting, and several modules from sklearn for machine learning tasks.

The load_local_20newsgroups function is designed to load text data from two specified categories within a given directory. It iterates through the categories, reads all text files in each category, and stores the text data and corresponding labels in lists. The path to the local data directory is defined, and all available category names are retrieved. Using itertools.combinations, all possible pairs of categories are generated for further processing.

The train_and_evaluate_for_pair function handles the training and evaluation of the Naive Bayes model for each pair of categories. It first loads the data for the specified pair of categories using the load_local_20newsgroups function. If data is successfully loaded, the dataset is split into training and testing sets using train_test_split. A machine learning pipeline is created with TfidfVectorizer for text feature extraction and MultinomialNB for classification. The model is trained on the training set, and predictions are made on the test set. The function then prints the classification report, displaying metrics such as precision, recall, and F1-score, and visualizes the confusion matrix using seaborn.

Finally, the script iterates through the first five pairs of categories, calling the train_and_evaluate_for_pair function for each pair. This allows the model to be trained and evaluated on different combinations of categories, demonstrating its performance across various classification tasks. 

4. I am using Spyder with Python 3.11 to run my code. Code imports different libraries like os, glob, numpy, scikit-learn and pip. As long as python is installed and has access to those libraries, nothing else needs to be installed, code will import them. I didn't need to install anything, it's all default program from anaconda. One thing to make sure is that I downloaded zip file from canvas with all the data set. In my code it looks like this: 
# Path to  local 20 newsgroups data directory
data_dir = r'C:\Users\Zakaria\Desktop\20_newsgroups'

User needs to download those very same files from canvas assignment page and put in correct path for data_dir. Other than that, code by default will generate 5 models of pairs 1vs1. code will print performance metric for each model alongside confusion matrix for each. To create models with 3 vs 3, in this line:
 # Generate all possible pairs of categories
category_pairs = list(itertools.combinations(all_categories, 2))   change 2 with 3 so that we create 3 pairs and rest stays same. If we want more than 5 models, we need to adjust accordingly in this line:  for pair in category_pairs[:5]:  and if we take out :5 from there, then it will train all possible pair combinations you can get.