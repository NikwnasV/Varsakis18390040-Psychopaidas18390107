import requests
from bs4 import BeautifulSoup
import pandas as pd

# Prompt the user to enter the number of articles to parse
n = int(input("Please enter the number of articles you want to parse: "))

for i in range(n):
    # Prompt the user to enter the last part of the Wikipedia link
    article_name = input(f"Enter the last part of the Wikipedia URL for article {i + 1} (e.g., 'Python_(programming_language)'): ")
    url = f"https://en.wikipedia.org/wiki/{article_name}"

    try:
        # Fetch the page
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

        # Parse the page content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the first paragraph
        paragraphs = soup.find_all('p')
        parsed_paragraphs = [p.text.strip() for p in paragraphs if p.text.strip()]

        # Save paragraphs to a separate CSV file for each article
        file_name = f"wikipedia_article_{i + 1}.csv"
        df = pd.DataFrame({'Paragraph': parsed_paragraphs})
        df.to_csv(file_name, index=False, encoding='utf-8')

        print(f"Article {i + 1} has been saved to '{file_name}'")

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {url}: {e}")