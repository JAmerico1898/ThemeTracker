import streamlit as st
import praw
import prawcore
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from textblob import TextBlob
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Download all required NLTK data in a proper try/except block
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')
except Exception as e:
    st.error(f"Error downloading NLTK data: {str(e)}")

class RedditThemeAnalyzer:
    def __init__(self):
        try:
            client_id = st.secrets.get('REDDIT_CLIENT_ID', os.environ.get('REDDIT_CLIENT_ID'))
            if not client_id:
                st.error("No API key found. Please configure the Reddit API key.")
                return
                
            client_secret = st.secrets.get('REDDIT_CLIENT_SECRET', os.environ.get('REDDIT_CLIENT_SECRET'))
            if not client_secret:
                st.error("No API secret found. Please configure the Reddit API secret.")
                return
                
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent="python:ThemesTracker:v1.0 (by /u/JAmerico1898)",
                username=st.secrets.get('REDDIT_USERNAME', os.environ.get('REDDIT_USERNAME')),
                password=st.secrets.get('REDDIT_PASSWORD', os.environ.get('REDDIT_PASSWORD'))
            )
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            raise

    def fetch_subreddit_posts(self, subreddit_name, time_filter='week', limit=100):
        """Fetch posts from specified subreddit within the last week."""
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = []
            
            for post in subreddit.top(time_filter=time_filter, limit=limit):
                try:
                    posts.append({
                        'title': post.title,
                        'text': post.selftext,
                        'score': post.score,
                        'comments': post.num_comments,
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'url': post.url
                    })
                except Exception as e:
                    st.warning(f"Error processing post: {str(e)}")
                    continue
                    
            return pd.DataFrame(posts)
            
        except prawcore.exceptions.ResponseException as e:
            st.error(f"Authentication Error: Please check your Reddit API credentials. Error: {str(e)}")
            return pd.DataFrame()
        except prawcore.exceptions.OAuthException as e:
            st.error(f"OAuth Error: Please verify your username and password. Error: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return pd.DataFrame()

    def preprocess_text(self, text):
        """Preprocess text by tokenizing, removing stopwords, and lemmatizing."""
        try:
            if not isinstance(text, str):
                text = str(text)
            
            # Handle empty or None text
            if not text or text.isspace():
                return []
                
            # Tokenize
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords and non-alphabetic tokens
            tokens = [token for token in tokens if token.isalpha() 
                     and token not in self.stop_words 
                     and len(token) > 1]  # Remove single-character tokens
            
            # Lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            return tokens
            
        except Exception as e:
            st.warning(f"Error preprocessing text: {str(e)}")
            return []

    def extract_themes(self, df):
        """Extract main themes from posts using NLP techniques."""
        try:
            # Combine title and text
            df['full_text'] = df['title'] + ' ' + df['text'].fillna('')
            
            # Process all texts
            all_tokens = []
            for text in df['full_text']:
                tokens = self.preprocess_text(text)
                all_tokens.extend(tokens)
            
            # Get most common terms
            word_freq = Counter(all_tokens)
            
            # Extract bigrams for more context
            bigrams = list(nltk.bigrams(all_tokens))
            bigram_freq = Counter(bigrams)
            
            return word_freq, bigram_freq
        except Exception as e:
            st.error(f"Error extracting themes: {str(e)}")
            return Counter(), Counter()

    def analyze_sentiment(self, df):
        """Analyze sentiment of posts to understand emotional context."""
        try:
            sentiments = []
            for text in df['full_text']:
                blob = TextBlob(str(text))
                sentiments.append(blob.sentiment.polarity)
            return pd.Series(sentiments)
        except Exception as e:
            st.error(f"Error analyzing sentiment: {str(e)}")
            return pd.Series([0] * len(df))

def main():
    st.title("Philosophical Themes Analyzer")
    st.write("Analyze trending themes from philosophy-related subreddits")

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    subreddits = st.sidebar.multiselect(
        "Select Subreddits",
        ["Existentialism", "SelfImprovement", "LifePurpose"],
        default=["Existentialism"]
    )
    
    time_filter = st.sidebar.selectbox(
        "Time Period",
        ["week", "month", "year"],
        index=0
    )
    
    try:
        analyzer = RedditThemeAnalyzer()
        
        if st.button("Analyze Themes"):
            with st.spinner("Analyzing subreddits..."):
                # Store results for all subreddits
                all_results = {}
                
                for subreddit in subreddits:
                    # Fetch and analyze posts
                    df = analyzer.fetch_subreddit_posts(subreddit, time_filter)
                    
                    if not df.empty:
                        word_freq, bigram_freq = analyzer.extract_themes(df)
                        sentiment = analyzer.analyze_sentiment(df)
                        
                        all_results[subreddit] = {
                            'posts': df,
                            'word_freq': word_freq,
                            'bigram_freq': bigram_freq,
                            'sentiment': sentiment.mean()
                        }
                
                # Display results
                if all_results:
                    st.header("Analysis Results")
                    
                    # Theme suggestions based on analysis
                    st.subheader("Suggested Speech Themes")
                    for subreddit, results in all_results.items():
                        st.write(f"\nThemes from r/{subreddit}:")
                        top_words = dict(sorted(results['word_freq'].items(), 
                                              key=lambda x: x[1], reverse=True)[:10])
                        
                        # Create word cloud
                        if top_words:
                            wordcloud = WordCloud(width=800, height=400, 
                                               background_color='white').generate_from_frequencies(top_words)
                            
                            fig, ax = plt.subplots()
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)
                        
                        # Display top bigrams for context
                        st.write("Common Phrases:")
                        top_bigrams = dict(sorted(results['bigram_freq'].items(), 
                                                key=lambda x: x[1], reverse=True)[:5])
                        for bigram, freq in top_bigrams.items():
                            st.write(f"- '{bigram[0]} {bigram[1]}' (mentioned {freq} times)")
                        
                        # Display sentiment analysis
                        st.write(f"Overall sentiment: {'Positive' if results['sentiment'] > 0 else 'Negative'} ({results['sentiment']:.2f})")
                
                # Generate speech titles based on analysis
                st.header("Suggested Speech Titles")
                
                def generate_titles(results_dict):
                    # Extract common words and phrases across all subreddits
                    all_words = []
                    all_bigrams = []
                    overall_sentiment = 0
                    
                    for sub_results in results_dict.values():
                        words = list(sub_results['word_freq'].keys())[:5]
                        all_words.extend(words)
                        bigrams = list(sub_results['bigram_freq'].keys())[:3]
                        all_bigrams.extend(bigrams)
                        overall_sentiment += sub_results['sentiment']
                    
                    # Generate titles based on analysis
                    titles = [
                        f"The Quest for Meaning: {all_words[0].title()} as a Path to Purpose",
                        f"From {all_words[1].title()} to {all_words[2].title()}: A Journey of Self-Discovery",
                        f"Understanding {all_words[3].title()}: The Key to Personal Growth",
                        f"The Art of {all_bigrams[0][0].title()} {all_bigrams[0][1].title()}: A Philosophical Perspective",
                        f"Beyond {all_words[4].title()}: Finding Authentic Purpose in Modern Life",
                        f"The {all_bigrams[1][0].title()} {all_bigrams[1][1].title()} Method: Transforming Self-Understanding",
                        f"Existential Wisdom: {all_words[0].title()} in the Age of Uncertainty",
                        f"The Power of {all_words[2].title()}: Navigating Life's Big Questions",
                        f"{all_bigrams[2][0].title()} {all_bigrams[2][1].title()}: A Blueprint for Personal Excellence",
                        f"Mastering {all_words[1].title()}: The Path to Authentic Living"
                    ]
                    
                    return titles
                
                if all_results:
                    titles = generate_titles(all_results)
                    st.subheader("Recommended Speech Titles:")
                    for i, title in enumerate(titles, 1):
                        st.write(f"{i}. {title}")
                        
                    # Add theme categories
                    st.subheader("Theme Categories:")
                    st.write("**Life Purpose:**")
                    st.write("- Titles 1, 5, and 10 focus on finding and living with purpose")
                    st.write("**Existentialism:**")
                    st.write("- Titles 3, 7, and 8 explore existential themes and life's meaning")
                    st.write("**Self Improvement:**")
                    st.write("- Titles 2, 4, 6, and 9 address personal growth and transformation")
                    
                else:
                    st.warning("No data was retrieved from the selected subreddits. Please check your credentials and try again.")
                    
    except Exception as e:
        st.error(f"An error occurred while running the analysis: {str(e)}")

if __name__ == "__main__":
    main()