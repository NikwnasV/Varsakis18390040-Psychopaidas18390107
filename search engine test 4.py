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
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import time
import os

# Η συνάρτηση αυτή, διαβάζει ένα αρχείο και το επιστρέφει σε συμβολοσειρά.
def read_file(file_name: str):
    text_file = open(file_name, "r", encoding="utf8")
    data = text_file.read()
    text_file.close()
    return data

# CISI Dataset Προετοιμασία
def load_cisi_dataset():
    docs = []
    queries = []
    relevance = {}

    # Ανάγνωση εγγράφων
    with open("CISI.ALL", encoding="utf8") as f:
        current_doc = []
        for line in f:
            if line.startswith(".I"):
                if current_doc:
                    docs.append(" ".join(current_doc))
                    current_doc = []
            elif line.startswith(".X"):
                continue
            else:
                current_doc.append(line.strip())
        if current_doc:
            docs.append(" ".join(current_doc))

    # Ανάγνωση queries
    with open("CISI.QRY", encoding="utf8") as f:
        current_query = []
        for line in f:
            if line.startswith(".I"):
                if current_query:
                    queries.append(" ".join(current_query))
                    current_query = []
            elif line.startswith(".W"):
                continue
            else:
                current_query.append(line.strip())
        if current_query:
            queries.append(" ".join(current_query))

    # Ανάγνωση relevance
    with open("CISI.REL", encoding="utf8") as f:
        for line in f:
            parts = line.strip().split()
            query_id = int(parts[0]) - 1
            doc_id = int(parts[1]) - 1
            if query_id not in relevance:
                relevance[query_id] = set()
            relevance[query_id].add(doc_id)

    return docs, queries, relevance

# Μονοπάτι που υπάρχουν τα αρχεία
path = "."
# Αποτέλεσμα αναζήτησεως αρχείων με κατάληξη txt εντός του path
rslt = [x for x in os.listdir(path) if x.endswith(".csv")]

# Φόρτωση δεδομένων
cisi_docs, cisi_queries, cisi_relevance = load_cisi_dataset()

# Δημιουργία dataframe
df = pd.DataFrame({"Title": [f"Doc_{i+1}" for i in range(len(cisi_docs))], "Text": cisi_docs})

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

# Αξιολόγηση

def evaluate_system(query_results, relevance, top_k=5):
    precision_scores = []
    recall_scores = []
    f1_scores = []
    average_precision_scores = []

    for query_id, retrieved_docs in enumerate(query_results):
        if query_id not in relevance:
            continue

        relevant_docs = relevance[query_id]
        retrieved_docs = set(retrieved_docs[:top_k])

        tp = len(retrieved_docs & relevant_docs)
        fp = len(retrieved_docs - relevant_docs)
        fn = len(relevant_docs - retrieved_docs)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        # Average Precision
        sorted_retrieved = list(retrieved_docs)
        ap = 0
        relevant_found = 0
        for i, doc_id in enumerate(sorted_retrieved):
            if doc_id in relevant_docs:
                relevant_found += 1
                ap += relevant_found / (i + 1)
        ap /= len(relevant_docs) if relevant_docs else 1
        average_precision_scores.append(ap)

    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)
    mean_ap = np.mean(average_precision_scores)

    return mean_precision, mean_recall, mean_f1, mean_ap

# Αποτελέσματα
bm25_results = [bm25.get_top_n(query.lower().split(), range(len(df)), n=10) for query in cisi_queries]
mean_precision, mean_recall, mean_f1, mean_ap = evaluate_system(bm25_results, cisi_relevance)

print(f"BM25 Evaluation:\nPrecision: {mean_precision:.3f}\nRecall: {mean_recall:.3f}\nF1-Score: {mean_f1:.3f}\nMAP: {mean_ap:.3f}")
