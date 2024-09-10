# Import necessary libraries
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

# Load the dataset
dataset = pd.read_csv("heart.csv")

# Display basic information about the dataset
print(dataset.info())
print(dataset.describe())
print(dataset.head())

# Visualize the target distribution
sns.countplot(x='target', data=dataset)
plt.title('Distribution of Heart Disease')
plt.show()

# Split the dataset into predictors and target
X = dataset.drop("target", axis=1)
y = dataset["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Initialize a list to store accuracy scores
scores = []

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
score_lr = round(accuracy_score(y_pred_lr, y_test) * 100, 2)
scores.append(score_lr)
print(f"Logistic Regression Accuracy: {score_lr}%")

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
score_nb = round(accuracy_score(y_pred_nb, y_test) * 100, 2)
scores.append(score_nb)
print(f"Naive Bayes Accuracy: {score_nb}%")

# Support Vector Machine
sv = svm.SVC(kernel='linear')
sv.fit(X_train, y_train)
y_pred_svm = sv.predict(X_test)
score_svm = round(accuracy_score(y_pred_svm, y_test) * 100, 2)
scores.append(score_svm)
print(f"Support Vector Machine Accuracy: {score_svm}%")

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
score_knn = round(accuracy_score(y_pred_knn, y_test) * 100, 2)
scores.append(score_knn)
print(f"K-Nearest Neighbors Accuracy: {score_knn}%")

# Decision Tree
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
score_dt = round(accuracy_score(y_pred_dt, y_test) * 100, 2)
scores.append(score_dt)
print(f"Decision Tree Accuracy: {score_dt}%")

# Random Forest
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
score_rf = round(accuracy_score(y_pred_rf, y_test) * 100, 2)
scores.append(score_rf)
print(f"Random Forest Accuracy: {score_rf}%")

# XGBoost
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
score_xgb = round(accuracy_score(y_pred_xgb, y_test) * 100, 2)
scores.append(score_xgb)
print(f"XGBoost Accuracy: {score_xgb}%")

# Neural Network
model = Sequential()
model.add(Dense(11, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=300, verbose=0)
y_pred_nn = model.predict(X_test)
y_pred_nn = [1 if x > 0.5 else 0 for x in y_pred_nn]
score_nn = round(accuracy_score(y_pred_nn, y_test) * 100, 2)
scores.append(score_nn)
print(f"Neural Network Accuracy: {score_nn}%")

# Summary of all models
algorithms = ["Logistic Regression", "Naive Bayes", "SVM", "KNN", "Decision Tree", "Random Forest", "XGBoost", "Neural Network"]
for i in range(len(algorithms)):
    print(f"The accuracy score achieved using {algorithms[i]} is: {scores[i]}%")

# Visualize the accuracy scores
sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")
sns.barplot(x=algorithms, y=scores)
plt.xticks(rotation=45)
plt.title("Model Comparison for Heart Disease Prediction")
plt.show()