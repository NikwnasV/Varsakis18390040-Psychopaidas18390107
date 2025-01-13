import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Ensure required NLTK data files are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools for text processing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Define a function to clean and process text
def process_text(text):
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Perform lemmatization
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]

    # Return the cleaned text as a single string
    return ' '.join(lemmatized)

# Process CSV files in the current directory
for file_name in os.listdir('.'):  # Change to the directory where your CSV files are stored
    if file_name.endswith('.csv'):
        print(f"Processing {file_name}...")

        # Read the CSV file
        df = pd.read_csv(file_name)

        # Ensure the CSV has a 'Paragraph' column
        if 'Paragraph' in df.columns:
            # Process each paragraph
            processed_data = []
            for paragraph in df['Paragraph']:
                processed = process_text(paragraph)
                processed_data.append({'Paragraph': processed})

            # Create a new DataFrame with processed data
            processed_df = pd.DataFrame(processed_data)

            # Save the processed data to a new CSV file
            processed_file_name = f"processed_{file_name}"
            processed_df.to_csv(processed_file_name, index=False, encoding='utf-8')
            print(f"Processed data saved to {processed_file_name}")
        else:
            print(f"Skipping {file_name}: 'Paragraph' column not found.")
