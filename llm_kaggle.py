import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

################################################ Data Preparation and Exploration #######################################################

# Load the dataset
try:
    df = pd.read_csv("C:/Master/classes/Machine learning/Kaggle_First/train.csv")
except FileNotFoundError:
    print("Error: 'train.csv' not found. Make sure you've downloaded the dataset.")
    exit()

# Explore the data
print(df.head())
print(df.info())

# Data Preprocessing: Create a binary winner column
df['winner'] = np.where(df['winner_model_a'] == 1, 'a', np.where(df['winner_model_b'] == 1, 'b', 'tie'))
df = df[df['winner'] != 'tie']  # Remove ties for binary classification

df['label'] = df['winner'].map({'a': 0, 'b': 1})  # Convert to numerical labels (a=0, b=1)
df['text'] = df['response_a'] + ' ' + df['response_b']  # Combine responses into one text column

df = df[['text', 'label']]  # Keep only relevant columns

# Check class balance after removing ties
print(f"Class distribution after removing ties:\n{df['label'].value_counts()}")

########################################### Feature Engineering ###################################################

# TF-IDF vectorization
print('Feature Engineering:')
vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
X = vectorizer.fit_transform(df['text'])
y = df['label']

############################################### Model Training and Evaluation ##########################################

# Split data into training and testing sets (80% training, 20% testing)
print('Model Training and Evaluation:')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model with Logistic Regression
model = LogisticRegression(solver='liblinear')  # 'liblinear' is suitable for small datasets
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model using accuracy and other metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')  # 5-fold cross-validation
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean CV accuracy: {np.mean(cv_scores)}")

# Hyperparameter Tuning
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}  # Regularization parameter tuning
grid_search = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(f"Best hyperparameters: {grid_search.best_params_}")

# Retrain model with the best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Make predictions with the tuned model
y_pred_best = best_model.predict(X_test)

# Evaluate the tuned model
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Tuned Model Accuracy: {accuracy_best}")
print("Tuned Model Classification Report:")
print(classification_report(y_test, y_pred_best))

########################################### Prediction on New Data ################################################

# Example of predicting on new data
new_responses = ["Response A text here", "Response B text here"]
new_X = vectorizer.transform(new_responses)
new_predictions = best_model.predict(new_X)  # Use the tuned model for prediction

print(f"Predictions for new responses: {new_predictions}")  # 0 indicates response A preferred, 1 indicates B

########################################### Save the Model ################################################

# Save the trained model and vectorizer
joblib.dump(best_model, 'logistic_regression_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved to disk.")

########################################### Kaggle Submission ################################################

# Prepare the test data
test_df = pd.read_csv('C:/Master/classes/Machine learning/Kaggle_First/test.csv')

# Preprocess the test data
test_text = test_df['response_a'] + ' ' + test_df['response_b']
test_X = vectorizer.transform(test_text)

# Make predictions on the test data
test_predictions = best_model.predict(test_X)

# Prepare the submission DataFrame
submission_df = pd.DataFrame({
    'id': test_df['id'],  # Assuming 'id' column exists in the test set
    'label': test_predictions
})

# Save the submission file in the required format (CSV)
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created.")
