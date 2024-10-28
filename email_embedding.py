import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Load the Word2Vec model
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

# Define text cleaning functions
def remove_unnecessary_words(s):
    tokenized_s = word_tokenize(s.lower())  # Convert to lowercase and tokenize
    add_to_stopwords = ['items', 'item', 'mails', 'mail', 'inboxes', 'inbox', 'threads', 'thread']
    to_remove = stopwords.words('english') + add_to_stopwords
    new_sent = [c for c in tokenized_s if c.isalnum() and c not in to_remove]
    return ' '.join(new_sent)

def remove_punctuation(s):
    return ''.join([char for char in s if char not in string.punctuation])

# Define vector embedding functions
def vector_rep(word):
    if word in model:
        return model[word]
    return np.zeros(300)

def general_vector_rep(phrase):
    tokenized = word_tokenize(phrase)
    if not tokenized:
        return np.zeros(300)
    vectors = [vector_rep(word) for word in tokenized if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)

# Load the CMU CERT email data
email_df = pd.read_csv("email_conc.csv")

# Clean the email content and apply the embedding
email_df['content_cleaned'] = email_df['content'].apply(remove_unnecessary_words).apply(remove_punctuation)
email_df['content_vector'] = email_df['content_cleaned'].apply(general_vector_rep)

# Convert the embedded vectors into separate columns (c0, c1, ..., c299)
embedded_df = pd.DataFrame(email_df['content_vector'].tolist(), columns=[f'c{i}' for i in range(300)])

# Concatenate the original email data (minus 'Content') with the new embedded columns
email_df = pd.concat([email_df.drop(['content', 'content_cleaned'], axis=1), embedded_df], axis=1)

# Save to a new CSV file
email_df.to_csv("http_with_embeddings.csv", index=False)
