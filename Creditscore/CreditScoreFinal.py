# Credit Score Check with Neural Network models, Logistic Regression, Random Forest
# Two files one to check accuracy of models and second is to predict the score using Random Forest
# Models accuracy is depends on correlation between data, correlation must be above around 0.5 score
# Kreggle for the dataset, and Excel function to generate new data.

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from warnings import filterwarnings

filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf

df = pd.read_csv(r"creditscore.csv")
ds = pd.read_csv(r"creditscorepred.csv")

df.head()
df.describe()

# Pie chart for Education
count = df['Education'].value_counts()
plt.pie(count, autopct='%1.1f%%', startangle=90)
plt.legend(labels=count.index, title=" Education Category", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
           edgecolor="black")
plt.title("\n Education Degree Percentage", fontsize=24)
plt.show()

# check column and row number
print("\ncheck column and row number")
print(df.shape)

# Check column head / names
print("\nCheck column head / names")
print(df.columns)

# save only non duplicate values

df = df.loc[~df.duplicated(subset=['Age', 'Gender', 'Income', 'Education', 'Marital Status',
                                   'Number of Children', 'Home Ownership'])].reset_index(drop=True).copy()

# Check column and row number
print("\nAfter remove duplicates, Check column and row number")
print(df.shape)

# check if duplicate values
print(df.loc[df.duplicated(subset=['Age', 'Gender', 'Income', 'Education', 'Marital Status',
                                   'Number of Children', 'Home Ownership'])])

# Correlation map
df['Gender'] = df['Gender'].astype('category').cat.codes
df['Education'] = df['Education'].astype('category').cat.codes
df['Marital Status'] = df['Marital Status'].astype('category').cat.codes
df['Home Ownership'] = df['Home Ownership'].astype('category').cat.codes
df['Credit Score'] = df['Credit Score'].astype('category').cat.codes
print(df.corr())

corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()


# setting values for model
X = df.drop('Credit Score', axis=1)
y = df['Credit Score']
X = pd.get_dummies(X, prefix=['Gender', 'Education', 'Marital Status', 'Home Ownership'],
                   columns=['Gender', 'Education', 'Marital Status', 'Home Ownership'])

# splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Logistic regression and Random Forest Models
lgr = LogisticRegression()
rf = RandomForestClassifier()

# Setting up Parameter values for models
lgr_param = {'C': [0.1, 1, 0.9]}
rf_param = {'n_estimators': [200, 300, 500], 'min_samples_split': [2, 4, 6]}

# use Grid Search to use different Parameter values
lgr_grid_search = GridSearchCV(lgr, param_grid=lgr_param, cv=5, scoring='accuracy')
rf_grid_search = GridSearchCV(rf, param_grid=rf_param, cv=5, scoring='accuracy')

# Train Both models
lgr_grid_search.fit(X_train, y_train)
rf_grid_search.fit(X_train, y_train)

# Print the Scores for models
print("\nLogistic Regression - Best Parameters:", lgr_grid_search.best_params_)
print("\nLogistic Regression - Best Score:", lgr_grid_search.best_score_)
print("\nRandom Forest - Best Parameters:", rf_grid_search.best_params_)
print("\nRandom Forest - Best Score:", rf_grid_search.best_score_)

# Apply best values to prediction
lgr_best_model = lgr_grid_search.best_estimator_
lgr_predictions = lgr_best_model.predict(X_test)

rf_best_model = rf_grid_search.best_estimator_
rf_predictions = rf_best_model.predict(X_test)

# Cal Accuracy
lgr_score = accuracy_score(y_test, lgr_predictions)
rf_score = accuracy_score(y_test, rf_predictions)

print("\n Logistic Regression Model Accuracy Score", lgr_score * 100)
print("\n Random Forest Model Accuracy Score", rf_score * 100)

# Now use random forest model to predict Credit Score for new data
final = ds.drop('Credit Score', axis=1)
final1 = ds['Credit Score']

final = pd.get_dummies(final, prefix=['Gender', 'Education', 'Marital Status', 'Home Ownership'],
                       columns=['Gender', 'Education', 'Marital Status', 'Home Ownership'], drop_first=True)
rf_grid_search.fit(final, final1)

rf_best_model = rf_grid_search.best_estimator_
final_pred = rf_best_model.predict(final)

# loop is generate the Score bese on model
print("\nPredict Credit Score for new data")
for index, tuples in ds.iterrows():
    print(f"Age: {tuples['Age']} Income: {tuples['Gender']} and {tuples['Home Ownership']}")
    print(f"Score:{final_pred[index]}")


# Neural Network with Tensorflow
X1 = df.drop('Credit Score', axis=1)
y1 = df['Credit Score']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2)
nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')])

# Compile model with parameter
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
# verbose don't show each iteration
nn_model.fit(X1_train, y1_train, epochs=400, batch_size=32, verbose=0)

# Evaluate Model
accuracy = nn_model.evaluate(X1_test, y1_test)

nn_pred = nn_model.predict(X1_test)

# Cal Accuracy
nn_score = accuracy_score(y1_test, nn_pred)
print("\n Neural Network Accuracy Score ", nn_score)

# print all score
print("\n Classification Report")
print(classification_report(y1_test, nn_pred))
