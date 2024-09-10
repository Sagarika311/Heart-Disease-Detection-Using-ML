# Heart Disease Prediction Using Machine Learning

This project implements various machine learning algorithms to predict the presence of heart disease based on patient data. The dataset used is the Heart Disease dataset, which contains various health metrics of patients, including age, sex, chest pain type, and other relevant features.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Models Used](#models-used)
- [Results](#results)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, you need to have Python installed along with the following libraries:

- pandas
- seaborn
- matplotlib
- scikit-learn
- xgboost
- keras

You can install the required libraries using pip:

pip install pandas seaborn matplotlib scikit-learn xgboost keras

## Usage

Clone this repository or download the files.
Place the heart.csv dataset in the same directory as the Python script.
Run the Python script:

python heart_disease_prediction.py

## Code Explanation

### Importing Libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense

The necessary libraries for data manipulation, visualization, and machine learning are imported.

### Loading the Dataset

dataset = pd.read_csv("heart.csv")

The dataset is loaded into a pandas DataFrame for analysis.

### Data Exploration

print(dataset.info())
print(dataset.describe())
print(dataset.head())

Basic information about the dataset, including data types and statistics, is displayed.

### Data Visualization

sns.countplot(x='target', data=dataset)
plt.title('Distribution of Heart Disease')
plt.show()

A count plot visualizes the distribution of the target variable (presence of heart disease).

### Data Preprocessing

X = dataset.drop("target", axis=1)
y = dataset["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

The dataset is split into features (X) and target (y), and then into training and testing sets.

### Model Training and Evaluation

The following models are trained and evaluated:
1. Logistic Regression
2. Naive Bayes
3. Support Vector Machine
4. K-Nearest Neighbors
5. Decision Tree
6. Random Forest
7. XGBoost
8. Neural Network

Each model is trained on the training data and evaluated on the test data. The accuracy of each model is printed.

Example Code for Model Training:
Logistic Regression:
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
score_lr = round(accuracy_score(y_pred_lr, y_test) * 100, 2)
print(f"Logistic Regression Accuracy: {score_lr}%")

### Results

The accuracy scores for all models are summarized and printed.

scores = [score_lr, score_nb, score_svm, score_knn, score_dt, score_rf, score_xgb, score_nn]
algorithms = ["Logistic Regression", "Naive Bayes", "SVM", "KNN", "Decision Tree", "Random Forest", "XGBoost", "Neural Network"]

for i in range(len(algorithms)):
    print(f"The accuracy score achieved using {algorithms[i]} is: {scores[i]}%")

### Visualization

sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")
sns.barplot(x=algorithms, y=scores)
plt.xticks(rotation=45)
plt.title("Model Comparison for Heart Disease Prediction")
plt.show()

A bar plot visualizes the accuracy scores of the different models.

## Summary

This README file provides a structured overview of your heart disease prediction project, including installation instructions, usage, code explanations, and results. You can customize it further based on specific details or additional features of your project. If you have any further questions or need additional help, feel free to ask!

## Contributing
Contributions are welcome! If you have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License
