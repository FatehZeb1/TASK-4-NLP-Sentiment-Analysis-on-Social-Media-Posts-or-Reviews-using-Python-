### Title: Sentiment Analysis on Social Media Posts or Reviews using Python

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample data (replace with your dataset)
data = {
    'review': [
        'I love this product! It is amazing.',
        'Terrible experience, would not recommend.',
        'Just okay, not great but not terrible either.',
        'Absolutely fantastic! Exceeded my expectations.',
        'Worst purchase ever. Very disappointed.'
    ],
    'sentiment': [1, 0, 1, 1, 0]  # 1 = Positive, 0 = Negative
}
df = pd.DataFrame(data)

# Preprocessing the data
def preprocess_text(text):
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    words = text.split()
    # Remove stopwords and stem the words
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words if word not in set(stopwords.words('english'))]
    # Join the words back into a single string
    return ' '.join(words)

df['review'] = df['review'].apply(preprocess_text)

# Extract features and labels
X = df['review']
y = df['sentiment']

# Convert text data into numerical data using CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(X).toarray()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy * 100}%')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
