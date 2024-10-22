import pandas as pd
import numpy as np
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import re
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import matplotlib.pyplot as plt
from tqdm import tqdm
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import grangercausalitytests
from Sentiment_Analysis_Dict import equity_markets_dict, query_dict


##import the dictionary from a seperate file, talk about how it involves my own judgement

class SentimentAnalyzer:
    def __init__(self, api_key: str, equity_market: str = "NASDAQ", etf: str = None,
                 llm_model: str = "distilbert-base-uncased-finetuned-sst-2-english", 
                 days_collected: int = 30, rolling_window: int = 7, 
                 classify_recession: bool = True, batch_size: int = 16):
        """                                                                                                                                                                                                         
        Initializes the SentimentAnalyzer with API key, equity market, model, days collected and the rolling window of the sentiment averaging.

          Args:
            api_key (str): API key for News API.
            equity_market (str): The equity market to analyze (default: "NASDAQ").
            etf (str): The specific ETF to use for analysis (optional).
            llm_model (str): Pre-trained model name for sentiment analysis (default: DistilBert).
            days_collected (int): Number of days to collect articles (default: 30).
            rolling_window (int): Size of the rolling window for sentiment averaging (default: 7).
            prompt_recession (bool): Whether to focus on recession fears in sentiment analysis (default: True).
            batch_size (int): Number of articles to process in each batch (default: 16).
            test_size (float): the size of the test set in the random forest model 
            random_state (int): random state of the test/train set aswell as the estimators of the random forest
        """
        
        self.api_key = api_key
        self.equity_market = equity_market
        self.etf = etf if etf else self.get_default_etf(self.equity_market)
        self.llm_model = llm_model
        self.days_collected = days_collected
        self.rolling_window = rolling_window
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classify_recession = classify_recession
        # Initialize the News API client
        self.newsapi = NewsApiClient(api_key=API_KEY)

        # Initialize sentiment analysis model
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.llm_model)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.llm_model)

    def fetch_articles(self, date, equity_market) -> list:
        """
        Fetches articles for a given date from the News API.

        Args:
            date (datetime): The date to fetch articles for.
            equity_market (str): The equity market for filtering articles.

        Returns:
            list: A list of articles fetched from the API. If an error occurs, returns an empty list.
        """
        try:
            date_str = date.strftime('%Y-%m-%d')
            query = query_dict[equity_market]
            articles = self.newsapi.get_everything(q=query,
                                            from_param=date_str,
                                            to=date_str,
                                            language='en',
                                            sort_by='relevancy')
            return articles['articles'] #see this datatype
        except Exception as e:
            print(f'Error fetching articles: {e}')
            return []
    
    def collect_articles(self) -> None:
        """
        Collects articles for the specified number of days and stores them in a DataFrame.

        This method iterates over the last 'days_collected' days, fetching articles for each day
        and appending them to a list. The collected articles are then converted into a pandas DataFrame.
        """
        all_articles = [] #better way to do this to lower memory
         # get data over last self days collected
        for i in range(self.days_collected):
            date = datetime.now() - timedelta(days=i)
            articles = self.fetch_articles(date, self.equity_market)
            all_articles.extend(articles)

        self.articles_df = pd.DataFrame(all_articles)
       
    
    def clean_articles(self) -> None:
        """
        Cleans the articles' title, description, and content for sentiment analysis.

        This method applies text preprocessing to remove unwanted characters and format the text
        into a consistent lowercase format, preparing the data for subsequent sentiment analysis.
        """
        def clean_text(text) -> str:
            ## Pre-process the text data
            if isinstance(text, str):
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                text = text.lower().strip()
            else:
                text = ""
            return text

        # Clean the title, description, and content
        self.articles_df['cleaned_title'] = self.articles_df['title'].apply(clean_text)
        self.articles_df['cleaned_description'] = self.articles_df['description'].apply(clean_text)
        self.articles_df['cleaned_content'] = self.articles_df['content'].apply(clean_text)

    def get_sentiment_batch(self, titles: list, descriptions: list, contents: list) -> list:
            """
            Processes batches of titles, descriptions, and contents to analyze sentiment with prompt engineering.

            Args:
                titles (list): List of article titles.
                descriptions (list): List of article descriptions.
                contents (list): List of article contents.

            Returns:
                tuple: Probabilities for titles, descriptions, and contents.
            """
            # Create prompts for each type of text
            if self.classify_recession:
                prompts_titles = [f"Analyze the following title for sentiment regarding recession fears:\n\n{title}" for title in titles]
                prompts_descriptions = [f"Analyze the following description for sentiment regarding recession fears:\n\n{description}" for description in descriptions]
                prompts_contents = [f"Analyze the following content for sentiment regarding recession fears:\n\n{content}" for content in contents]
            else: #ge model perfomance
                etf_description = equity_markets_dict[self.equity_market]["etf_description"]
                prompts_titles = [f"Analyze the following title for sentiment regarding the performance of {self.equity_market}, which {etf_description}:\n\n{title}" for title in titles]
                prompts_descriptions = [f"Analyze the following description for sentiment regarding the performance of {self.equity_market}, which {etf_description}:\n\n{description}" for description in descriptions]
                prompts_contents = [f"Analyze the following content for sentiment regarding the performance of {self.equity_market}, which {etf_description}:\n\n{content}" for content in contents]

            # Tokenize all three types of texts
            inputs_titles = self.tokenizer(prompts_titles, return_tensors="pt", padding=True, truncation=True).to(self.device)
            inputs_descriptions = self.tokenizer(prompts_descriptions, return_tensors="pt", padding=True, truncation=True).to(self.device)
            inputs_contents = self.tokenizer(prompts_contents, return_tensors="pt", padding=True, truncation=True).to(self.device)

            with torch.no_grad():
                logits_titles = self.model(**inputs_titles).logits
                logits_descriptions = self.model(**inputs_descriptions).logits
                logits_contents = self.model(**inputs_contents).logits

            # Get probabilities
            probs_titles = torch.nn.functional.softmax(logits_titles, dim=1).cpu().numpy()
            probs_descriptions = torch.nn.functional.softmax(logits_descriptions, dim=1).cpu().numpy()
            probs_contents = torch.nn.functional.softmax(logits_contents, dim=1).cpu().numpy()

            # Classify sentiments
            if self.classify_recession:
                classifications_titles = self.classify_sentiment(probs_titles)
                classifications_descriptions = self.classify_sentiment(probs_descriptions)
                classifications_contents = self.classify_sentiment(probs_contents)

                return probs_titles, probs_descriptions, probs_contents, classifications_titles, classifications_descriptions, classifications_contents

            else:
                return probs_titles, probs_descriptions, probs_contents
    
    def classify_sentiment(self, probs):
            """
            Classifies sentiment based on probabilities.

            Args:
                probs (numpy.ndarray): Array of probabilities for each sentiment class.

            Returns:
                list: Sentiment classifications with confidence scores.
            """
            classifications = []
            for prob in probs:
                if prob[1] > 0.5:  # Positive sentiment
                    classifications.append(('Positive', prob[1]))
                elif prob[0] > 0.5:  # Negative sentiment
                    classifications.append(('Negative', prob[0]))
                else:  # Neutral sentiment
                    classifications.append(('Neutral', max(prob)))

            return classifications
    
   
    


              
    def sentiment_of_articles(self) -> None:
        """
        Analyzes the sentiment of the collected articles and stores the results in a DataFrame.

        This method processes the articles in batches to efficiently analyze sentiment and
        collects the results into a structured DataFrame for further analysis.
        """
        # Prepare lists for batch processing
        titles = self.articles_df['cleaned_title'].tolist()
        descriptions = self.articles_df['cleaned_description'].tolist()
        contents = self.articles_df['cleaned_content'].tolist()

        sentiment_results = []


        # Process all articles in batches
        for i in tqdm(range(0, len(titles), self.batch_size), desc="Analyzing articles"):
            batch_titles = titles[i:i + self.batch_size]
            batch_descriptions = descriptions[i:i + self.batch_size]
            batch_contents = contents[i:i + self.batch_size]

            if self.classify_recession:
                # Get sentiment classifications for the batch
                title_probs, description_probs, content_probs, classifications_titles, classifications_descriptions, classifications_contents = self.get_sentiment_batch(batch_titles, batch_descriptions, batch_contents)
                
                # Collect results
                for j in range(len(title_probs)):
                    sentiment_results.append({
                        'date': self.articles_df['publishedAt'][i + j][:10],  # Ensure correct date mapping
                        'positive_prob_title': title_probs[j][1],
                        'negative_prob_title': title_probs[j][0],
                        'positive_prob_description': description_probs[j][1],
                        'negative_prob_description': description_probs[j][0],
                        'positive_prob_content': content_probs[j][1],
                        'negative_prob_content': content_probs[j][0],
                        'sentiment_title': classifications_titles[j][0],
                        'confidence_title': classifications_titles[j][1],
                        'sentiment_description': classifications_descriptions[j][0],
                        'confidence_description': classifications_descriptions[j][1],
                        'sentiment_content': classifications_contents[j][0],
                        'confidence_content': classifications_contents[j][1]
                    })
                        
                       
              

            else:
                # Get sentiment probabilities for the batch
                title_probs, description_probs, content_probs = self.get_sentiment_batch(batch_titles, batch_descriptions, batch_contents)

                # Collect results if 
            
                for j in range(len(title_probs)):
                    sentiment_results.append({
                        'date': self.articles_df['publishedAt'][i + j][:10],  # Ensure correct date mapping
                        'positive_prob_title': title_probs[j][1],
                        'negative_prob_title': title_probs[j][0],
                        'positive_prob_description': description_probs[j][1],
                        'negative_prob_description': description_probs[j][0],
                        'positive_prob_content': content_probs[j][1],
                        'negative_prob_content': content_probs[j][0]
                    })

            # Create a sentiment DataFrame
            self.sentiment_df = pd.DataFrame(sentiment_results)
        

    def collect_equity_data(self) -> None:
        """
        Collects equity market data for the specified market and date range.

        This method fetches the historical closing prices for the specified equity market 
        over the last 'days_collected' days using the yfinance library.
        """
        start_date = (datetime.now() - timedelta(days=self.days_collected)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Use the ETF for fetching data
        if self.etf:
            self.equity_data = yf.download(self.etf, start=start_date, end=end_date)
        else:
            self.equity_data = yf.download(self.equity_market, start=start_date, end=end_date)
        
        self.equity_data.reset_index(inplace=True)
        
    
 
    def calculate_correlation(self) -> tuple:
        '''COULD WRITE THIS BIT BETTER'''

        
        """
        Calculates correlation coefficients between sentiment scores and equity closing prices.

        Returns:
            tuple: Spearman and Pearson correlation coefficients for titles, descriptions, and content.
        """
        # Convert the 'date' column in daily_sentiment to datetime
        self.daily_sentiment['date'] = pd.to_datetime(self.daily_sentiment['date'])

        # Ensure equity_data['Date'] is in datetime format
        self.equity_data['Date'] = pd.to_datetime(self.equity_data['Date'])

        # Merge the DataFrames on the date columns
        merged_data = pd.merge(self.daily_sentiment, self.equity_data, left_on='date', right_on='Date', how='inner')

        # Drop any rows with NaN values that might have arisen from merging
        merged_data.dropna(inplace=True)

        # Calculate correlations
        spearman_corr_title_positive, _ = spearmanr(merged_data['rolling_sentiment_title_positive'], merged_data['Close'])
        pearson_corr_title_positive, _ = pearsonr(merged_data['rolling_sentiment_title_positive'], merged_data['Close'])

        spearman_corr_title_negative, _ = spearmanr(merged_data['rolling_sentiment_title_negative'], merged_data['Close'])
        pearson_corr_title_negative, _ = pearsonr(merged_data['rolling_sentiment_title_negative'], merged_data['Close'])

        spearman_corr_description_positive, _ = spearmanr(merged_data['rolling_sentiment_description_positive'], merged_data['Close'])
        pearson_corr_description_positive, _ = pearsonr(merged_data['rolling_sentiment_description_positive'], merged_data['Close'])

        spearman_corr_description_negative, _ = spearmanr(merged_data['rolling_sentiment_description_negative'], merged_data['Close'])
        pearson_corr_description_negative, _ = pearsonr(merged_data['rolling_sentiment_description_negative'], merged_data['Close'])

        spearman_corr_content_positive, _ = spearmanr(merged_data['rolling_sentiment_content_positive'], merged_data['Close'])
        pearson_corr_content_positive, _ = pearsonr(merged_data['rolling_sentiment_content_positive'], merged_data['Close'])

        spearman_corr_content_negative, _ = spearmanr(merged_data['rolling_sentiment_content_negative'], merged_data['Close'])
        pearson_corr_content_negative, _ = pearsonr(merged_data['rolling_sentiment_content_negative'], merged_data['Close'])
        
        # Granger causality tests
        max_lag = 4
        granger_results = {}
        
        # Assuming you want to test if sentiment scores can predict closing prices
        for sentiment_column in [
            'rolling_sentiment_title_positive',
            'rolling_sentiment_description_positive',
            'rolling_sentiment_content_positive',
            'rolling_sentiment_title_negative',
            'rolling_sentiment_description_negative',
            'rolling_sentiment_content_negative'
        ]:
            test_result = grangercausalitytests(merged_data[[sentiment_column, 'Close']], max_lag, verbose=False)
            granger_results[sentiment_column] = {
                lag: {'F-statistic': test_result[lag][0]['ssr_ftest'][0], 'p-value': test_result[lag][0]['ssr_ftest'][1]}
                for lag in range(1, max_lag + 1)
            }
            
        return (spearman_corr_title_positive, pearson_corr_title_positive, spearman_corr_description_positive, pearson_corr_description_positive, spearman_corr_content_positive, pearson_corr_content_positive,
                spearman_corr_title_negative, pearson_corr_title_negative, spearman_corr_description_negative, pearson_corr_description_negative, spearman_corr_content_negative, pearson_corr_content_negative,
                granger_results)


     
    def plot_equity(self) -> None:
        """
        Plots the sentiment trends and equity closing prices.

        This method visualizes the daily average sentiment scores against the closing prices
        of the specified equity market, helping to assess the relationship between sentiment and market performance.
        """
        # Aggregate daily sentiment scores
        self.daily_sentiment = self.sentiment_df.groupby('date').mean().reset_index()

        # Optional: Add previous days' context (e.g., rolling mean)
        self.daily_sentiment['rolling_sentiment_title_positive'] = self.daily_sentiment['positive_prob_title'].rolling(window=self.rolling_window).mean()
        self.daily_sentiment['rolling_sentiment_description_positive'] = self.daily_sentiment['positive_prob_description'].rolling(window=self.rolling_window).mean()
        self.daily_sentiment['rolling_sentiment_content_positive'] = self.daily_sentiment['positive_prob_content'].rolling(window=self.rolling_window).mean()

        self.daily_sentiment['rolling_sentiment_title_negative'] = self.daily_sentiment['negative_prob_title'].rolling(window=self.rolling_window).mean()
        self.daily_sentiment['rolling_sentiment_description_negative'] = self.daily_sentiment['negative_prob_description'].rolling(window=self.rolling_window).mean()
        self.daily_sentiment['rolling_sentiment_content_negative'] = self.daily_sentiment['negative_prob_content'].rolling(window=self.rolling_window).mean()

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(self.daily_sentiment['date'], self.daily_sentiment['rolling_sentiment_title_positive'], label='Sentiment Score title positive', color='blue')
        plt.plot(self.daily_sentiment['date'], self.daily_sentiment['rolling_sentiment_description_positive'], label='Sentiment Score description positive', color='red')
        plt.plot(self.daily_sentiment['date'], self.daily_sentiment['rolling_sentiment_content_positive'], label='Sentiment Score content positive', color='green')
        plt.plot(self.daily_sentiment['date'], self.daily_sentiment['rolling_sentiment_title_negative'], label='Sentiment Score title negative', color='black')
        plt.plot(self.daily_sentiment['date'], self.daily_sentiment['rolling_sentiment_description_negative'], label='Sentiment Score description negative', color='purple')
        plt.plot(self.daily_sentiment['date'], self.daily_sentiment['rolling_sentiment_content_negative'], label='Sentiment Score content negative', color='orange')

 
        # You can load and plot your index return data here
        # Example: plt.plot(sp500_df['date'], sp500_df['index_return'], label='S&P 500 Return', color='orange')

        plt.title('Sentiment Trends in Equity Markets')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        # Define the date range

        # Save the figure
        plt.savefig(f'{self.equity_market}_sentiment_trends.png', dpi=300)
        plt.close()  # Close the figure to free up memory


         # Fetch equity data for plotting
        self.collect_equity_data()
    
        # Plotting
        plt.figure(figsize=(14, 7))
        plt.plot(self.equity_data['Date'], self.equity_data['Close'], label=f'{self.equity_market} Close Price', color='blue')
        plt.xticks(rotation=45)
        plt.title(f'{self.equity_market} Closing Prices')
        plt.xlabel('Date')
        plt.ylabel(f'{self.equity_market} Close Price (USD)')
        plt.legend()
        plt.tight_layout()

        plt.savefig(f'{self.equity_market}_closing_prices.png', dpi=300)
        plt.close()  # Close the figure to free up memory

        # plot correlation matrix of sentiments from different surces, title, description and content
        correlation_matrix = self.daily_sentiment[['rolling_sentiment_title_positive', 
                                                'rolling_sentiment_description_positive', 
                                                'rolling_sentiment_content_positive',
                                                'rolling_sentiment_title_negative', 
                                                'rolling_sentiment_description_negative', 
                                                'rolling_sentiment_content_negative']].corr()

        # Plotting the correlation matrix
        plt.figure(figsize=(10, 5))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix of Sentiment")
        # Save the correlation matrix figure
        plt.savefig(f'{self.equity_market}_correlation_matrix.png', dpi=300)
        plt.close()  # Close the figure to free up memory
    
  



    def time_series_process(self) -> None: 
        """
        Executes the entire process of collecting articles, cleaning, analyzing sentiment, and plotting results.

        This method orchestrates all steps from data collection to visualization and correlation analysis,
        providing a comprehensive overview of sentiment trends related to equity markets.
        """
        self.collect_articles()
        self.clean_articles()
        self.sentiment_of_articles()
        self.plot_equity()
        (spearman_corr_title_positive, pearson_corr_title_positive, spearman_corr_description_positive, pearson_corr_description_positive, spearman_corr_content_positive, pearson_corr_content_positive,
                spearman_corr_title_negative, pearson_corr_title_negative, spearman_corr_description_negative, pearson_corr_description_negative, spearman_corr_content_negative, pearson_corr_content_negative, granger_results) = self.calculate_correlation()
        print(f'Spearman Correlation Positive (Title): {spearman_corr_title_positive:.4f}')
        print(f'Pearson Correlation Positive (Title): {pearson_corr_title_positive:.4f}')
        
        print(f'Spearman Correlation Positive (Description): {spearman_corr_description_positive:.4f}')
        print(f'Pearson Correlation Positive (Description): {pearson_corr_description_positive:.4f}')
        
        print(f'Spearman Correlation Positive (Content): {spearman_corr_content_positive:.4f}')
        print(f'Pearson Correlation Positive (Content): {pearson_corr_content_positive:.4f}')

        print(f'Spearman Correlation Negative (Title): {spearman_corr_title_negative:.4f}')
        print(f'Pearson Correlation Negative (Title): {pearson_corr_title_negative:.4f}')
        
        print(f'Spearman Correlation Negative (Description): {spearman_corr_description_negative:.4f}')
        print(f'Pearson Correlation Negative (Description): {pearson_corr_description_negative:.4f}')
        
        print(f'Spearman Correlation Negative (Content): {spearman_corr_content_negative:.4f}')
        print(f'Pearson Correlation Negative (Content): {pearson_corr_content_negative:.4f}')

        print(f'Granger test results: {granger_results}')

        

# Example usage
if __name__ == "__main__":
    API_KEY = 'fecccf92ed314ae1be49290cbb07d195'  
    sentiment_analyzer = SentimentAnalyzer(api_key = API_KEY, equity_market="XLB", etf='XLB', llm_model = "distilbert-base-uncased-finetuned-sst-2-english", days_collected = 30, rolling_window = 7, classify_recession=False, batch_size = 16)
    sentiment_analyzer.time_series_process()
    