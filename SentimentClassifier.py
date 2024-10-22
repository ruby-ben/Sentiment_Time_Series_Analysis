import pandas as pd
import numpy as np
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import re
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from Sentiment_Analysis_Dict import equity_markets_dict, query_dict

class SentimentClassifier:
    def __init__(self, api_key: str, equity_market: str = "NASDAQ", etf: str = "QQQ",
                 llm_model: str = "distilbert-base-uncased-finetuned-sst-2-english", 
                 days_collected: int = 30, batch_size: int = 16):
        """                                                                                                                                                                                                         
        Initializes the SentimentAnalyzer with API key, equity market, model, days collected and the rolling window of the sentiment averaging.

          Args:
            api_key (str): API key for News API.
            equity_market (str): The equity market to analyze (default: "NASDAQ").
            etf (str): The specific ETF to use for analysis (optional).
            llm_model (str): Pre-trained model name for sentiment analysis (default: DistilBert).
            days_collected (int): Number of days to collect articles (default: 30).
            batch_size (int): Number of articles to process in each batch (default: 16).
        """
        
        self.api_key = api_key
        self.equity_market = equity_market
        self.etf = etf
        self.llm_model = llm_model
        self.days_collected = days_collected
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the News API client
        self.newsapi = NewsApiClient(api_key=self.api_key)

        # Initialize sentiment analysis model
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.llm_model)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.llm_model)

    def fetch_articles(self, date, equity_market) -> list:
        """Fetches articles for a given date from the News API."""
        try:
            date_str = date.strftime('%Y-%m-%d')
        # Adjust this as needed for your specific query
            query = query_dict[equity_market]
            articles = self.newsapi.get_everything(q=query,
                                            from_param=date_str,
                                            to=date_str,
                                            language='en',
                                            sort_by='relevancy')
            return articles.get('articles', [])
        except Exception as e:
            print(f'Error fetching articles: {e}')
            return []

    def collect_articles(self) -> None:
        """Collects articles for the specified number of days and stores them in a DataFrame."""
        all_articles = []
        for i in range(self.days_collected):
            date = datetime.now() - timedelta(days=i)
            articles = self.fetch_articles(date, self.equity_market)
            all_articles.extend(articles)

        self.articles_df = pd.DataFrame(all_articles)

    def clean_articles(self) -> None:
        """Cleans the articles' title, description, and content for sentiment analysis."""
        def clean_text(text) -> str:
            if isinstance(text, str):
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                text = text.lower().strip()
            return text

        self.articles_df['cleaned_title'] = self.articles_df['title'].apply(clean_text)
        self.articles_df['cleaned_description'] = self.articles_df['description'].apply(clean_text)
        self.articles_df['cleaned_content'] = self.articles_df['content'].apply(clean_text)

    def get_sentiment_batch(self, texts: list) -> np.ndarray:
        """Analyzes sentiment for a batch of texts."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Get probabilities
        return torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

    def classify_sentiment(self, probs: np.ndarray) -> list:
        """Classifies sentiment based on probabilities."""
        classifications = []
        for prob in probs:
            if prob [1] > prob [0]:
                if prob[1] > 0.5:  # Positive sentiment
                    classifications.append(('Positive', prob[1]))
                else:  # Neutral sentiment
                    classifications.append(('Neutral', max(prob)))
            else:
                if prob[0] > 0.5:  # Negative sentiment
                    classifications.append(('Negative', prob[0]))
                else:  # Neutral sentiment
                    classifications.append(('Neutral', max(prob)))

        return classifications

    def analyze_sentiment(self) -> None:
        """Main method to collect, clean, and analyze sentiment of articles."""
        self.collect_articles()
        self.clean_articles()
        
        all_sentiments = []
        for index, row in self.articles_df.iterrows():
            texts = [
                row['cleaned_title'] or "",  # Default to empty string if None
                row['cleaned_description'] or "",
                row['cleaned_content'] or ""
            ]

            # Pass the texts list directly
            probs = self.get_sentiment_batch(texts)
            classifications = self.classify_sentiment(probs)
            all_sentiments.append(classifications)
        
        self.articles_df['sentiment'] = all_sentiments

# Example usage

# Example usage
if __name__ == "__main__":
    API_KEY = 'fecccf92ed314ae1be49290cbb07d195' 
    sentiment_classifier = SentimentClassifier(api_key=API_KEY)
    sentiment_classifier.analyze_sentiment()
    print(sentiment_classifier.articles_df[['title', 'sentiment']])