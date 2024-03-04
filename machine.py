# Importing necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# Sample data (replace this with your own dataset)
# X = ["I love this movie", "This movie is great", "I hate this movie", "This movie is terrible"]
# y = ["positive", "positive", "negative", "negative"]
from sample_data import sample_data
from labels import labels

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sample_data, labels, test_size=0.2, random_state=42)

# Feature extraction (converting text data into numerical features)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
# print(X_train_vectorized)
# Training the classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Making predictions
y_pred = classifier.predict(X_test_vectorized)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Example usage
userInput = input("Write a review: ")
# new_texts = ["I think this movie is amazing", "I did not enjoy this movie at all"]
userInput_vectorized = vectorizer.transform([userInput])
predictions = classifier.predict(userInput_vectorized)
print("Predictions:", predictions)
