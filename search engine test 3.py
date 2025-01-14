# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:06:55 2025

@author: Agisilaos
"""

import pandas as pd
import spacy
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import os

# Η συνάρτηση αυτή, διαβάζει ένα αρχείο και το επιστρέφει σε συμβολοσειρά.
def read_file(file_name:str):
    text_file = open(file_name, "r", encoding="utf8")
    data = text_file.read()
    text_file.close()
    return data

# Μονοπάτι που υπάρχουν τα αρχεία
path = "."
# Αποτέλεσμα αναζήτησεως αρχείων με κατάληξη .csv εντός του path
rslt = [x for x in os.listdir(path) if x.endswith(".csv")]

# Για κάθε ένα έγγραφο διάβασε το περιεχόμενο του
# και πρόσθεσε το στο dataframe μαζί με το όνομα του αρχείου ως τίτλο
rows = []
for file in rslt:
    text = read_file(path + '/' + file)
    title = file.split('.')
    row = {'Title': title[0], 'Text': text}
    rows.append(row)

# Δημιουργία κενού dataframe
df = pd.DataFrame(rows)

pd.set_option('display.max_colwidth', None)
nlp = spacy.load("en_core_web_sm")
tok_text = []

# Κάνε tokenize
for doc in tqdm(nlp.pipe(df.Text.str.lower().values, disable=["tagger", "parser", "ner"])):
    tok = [t.text for t in doc if t.is_alpha]
    tok_text.append(tok)

df['Tokenized'] = tok_text

# BM25
bm25 = BM25Okapi(tok_text)

# Vector Space Model (VSM)
def build_vsm_matrix(documents):
    unique_terms = list(set(term for doc in documents for term in doc))
    term_index = {term: idx for idx, term in enumerate(unique_terms)}
    matrix = np.zeros((len(documents), len(unique_terms)))

    for doc_idx, doc in enumerate(documents):
        for term in doc:
            matrix[doc_idx, term_index[term]] += 1

    # Normalize rows to unit vectors
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix, term_index

vsm_matrix, term_index = build_vsm_matrix(tok_text)

# Boolean Retrieval
inverted_index = {}
for idx, tokens in enumerate(tok_text):
    for token in set(tokens):
        if token not in inverted_index:
            inverted_index[token] = set()
        inverted_index[token].add(idx)

# Query Processing and Boolean Operations
def process_boolean_query(query, inverted_index):
    terms = query.lower().split()
    result_set = set(range(len(tok_text)))
    i = 0

    while i < len(terms):
        if terms[i] == "and":
            i += 1
            result_set &= inverted_index.get(terms[i], set())
        elif terms[i] == "or":
            i += 1
            result_set |= inverted_index.get(terms[i], set())
        elif terms[i] == "not":
            i += 1
            result_set -= inverted_index.get(terms[i], set())
        else:
            result_set &= inverted_index.get(terms[i], set())
        i += 1

    return result_set

# User Query
query = input('Search: ')
print("Choose the algorithm to use:")
print("1. BM25")
print("2. VSM")
print("3. Boolean Retrieval")
choice = input("Enter the number of your choice: ")

if choice == "1":
    tokenized_query = query.lower().split(" ")
    t0 = time.time()
    bm25_results = bm25.get_top_n(tokenized_query, df.Text.values, n=5)
    bm25_titles = bm25.get_top_n(tokenized_query, df.Title.values, n=5)
    t1 = time.time()
    print(f"BM25 searched {len(rslt)} records in {round(t1-t0, 3)} seconds \n")
    for text, title in zip(bm25_results, bm25_titles):
        print("BM25 File name: " + path + '/' + title + '.csv')
        print(text)

elif choice == "2":
    query_tokens = [token.text for token in nlp(query.lower()) if token.is_alpha]
    query_vector = np.zeros(vsm_matrix.shape[1])
    for token in query_tokens:
        if token in term_index:
            query_vector[term_index[token]] += 1

    # Normalize query vector
    query_vector = query_vector / np.linalg.norm(query_vector)

    t0 = time.time()
    similarities = cosine_similarity(query_vector.reshape(1, -1), vsm_matrix).flatten()
    top_indices = similarities.argsort()[-5:][::-1]
    t1 = time.time()

    print(f"VSM searched {len(rslt)} records in {round(t1-t0, 3)} seconds \n")
    for idx in top_indices:
        print("VSM File name: " + path + '/' + df.Title.values[idx] + '.csv')
        print(df.Text.values[idx])

elif choice == "3":
    if any(op in query.lower() for op in ["and", "or", "not"]):
        t0 = time.time()
        boolean_results = process_boolean_query(query, inverted_index)
        t1 = time.time()
        print(f"Boolean retrieval searched {len(rslt)} records in {round(t1-t0, 3)} seconds \n")
        for idx in boolean_results:
            print("Boolean Retrieval File name: " + path + '/' + df.Title.values[idx] + '.csv')
            print(df.Text.values[idx])
    else:
        print("Invalid Boolean query. Please use operators AND, OR, NOT correctly.")
