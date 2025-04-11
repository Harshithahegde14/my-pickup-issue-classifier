# import pandas as pd
# import re
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import make_pipeline
# import joblib

# # Step 1: Load Dataset
# df = pd.read_csv('my_pickup_full_dataset.csv')

# print(f"Original dataset shape: {df.shape}")

# # Step 2: Clean the data

# def clean_text(text):
#     text = text.lower()  # Lowercase
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
#     text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
#     return text

# df['message'] = df['message'].astype(str).apply(clean_text)

# # Drop any rows with missing values
# df.dropna(inplace=True)

# # Optional: Remove duplicates
# df.drop_duplicates(subset=['message', 'category'], inplace=True)

# print(f"Cleaned dataset shape: {df.shape}")

# # Step 3: Prepare features and labels
# X = df['message']
# y = df['category']

# # Step 4: Create and train the model
# model = make_pipeline(TfidfVectorizer(), MultinomialNB())
# model.fit(X, y)

# # Step 5: Save the model
# joblib.dump(model, 'model.pkl')

# print("✅ Model trained and saved as 'model.pkl' successfully!")

# # Optional: Save cleaned dataset (for future reference)
# df.to_csv('cleaned_dataset.csv', index=False)
# print("✅ Cleaned dataset saved as 'cleaned_dataset.csv'.")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re

# Load dataset
data = pd.read_csv('my_pickup_full_dataset.csv')

# Clean text
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Apply cleaning
data['message'] = data['message'].astype(str).apply(clean_text)

# Remove empty rows
data = data.dropna(subset=['message', 'department', 'category'])
data = data[data['message'].str.strip() != ""]

print(f"Original dataset shape: {data.shape}")

# Save cleaned data
data.to_csv('cleaned_dataset.csv', index=False)
print("✅ Cleaned dataset saved as 'cleaned_dataset.csv'.")

# Feature and target
X = data['message']
y = data['category']  # Predicting category

# Vectorization
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("✅ Model and vectorizer saved successfully!")
