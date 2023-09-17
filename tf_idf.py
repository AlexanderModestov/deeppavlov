from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Compute the TF-IDF representation
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Get feature names (terms/words) from the vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()

# Print the TF-IDF matrix
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())

# Print the feature names
print("\nFeature Names:")
print(feature_names)

# Optionally, if you want to transform a new document to its TF-IDF representation
new_document = "This is a new document."
tfidf_new_doc = tfidf_vectorizer.transform([new_document])

# Print the TF-IDF representation of the new document
print("\nTF-IDF representation of the new document:")
print(tfidf_new_doc.toarray())