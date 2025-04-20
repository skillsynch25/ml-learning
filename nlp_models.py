import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
from collections import Counter

class TextPreprocessor:
    """
    A comprehensive text preprocessing class that handles various NLP preprocessing tasks.
    
    This class implements several key preprocessing steps:
    1. Text Cleaning: Removes numbers, special characters, and converts to lowercase
    2. Tokenization: Splits text into words or sentences
    3. Stopword Removal: Removes common words that add little meaning
    4. Word Normalization: Reduces words to their base form through lemmatization or stemming
    
    Educational Notes:
    - Text preprocessing is crucial for NLP tasks as it reduces noise and standardizes text
    - Different preprocessing steps may be needed depending on the specific NLP task
    - The order of preprocessing steps can affect the final results
    """
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text, remove_numbers=True, remove_special_chars=True, lowercase=True):
        """
        Cleans text by removing numbers, special characters, and converting to lowercase.
        
        Parameters:
        - text: Input text to clean
        - remove_numbers: Whether to remove numerical digits
        - remove_special_chars: Whether to remove special characters
        - lowercase: Whether to convert text to lowercase
        
        Educational Notes:
        - Text cleaning helps standardize input data
        - Removing numbers may be important for some tasks but not others
        - Converting to lowercase helps reduce vocabulary size
        - Special characters may be important for some tasks (e.g., sentiment analysis)
        """
        if remove_numbers:
            text = text.str.replace(r'\d+', '')
        if remove_special_chars:
            text = text.str.replace(r'[^\w\s]', '')
        if lowercase:
            text = text.str.lower()
        return text
    
    def tokenize(self, text, method="word"):
        """
        Tokenizes text into words or sentences.
        
        Parameters:
        - text: Input text to tokenize
        - method: Tokenization method ("word", "sentence", or "both")
        
        Educational Notes:
        - Tokenization is the process of breaking text into smaller units
        - Word tokenization splits text into individual words
        - Sentence tokenization splits text into sentences
        - Different tokenization methods may be needed for different tasks
        """
        if method == "word":
            return text.str.split()
        elif method == "sentence":
            return text.str.split('.')
        else:  # both
            words = text.str.split()
            sentences = text.str.split('.')
            return pd.DataFrame({
                'words': words,
                'sentences': sentences
            })
    
    def remove_stopwords(self, tokens, custom_stopwords=None):
        """
        Removes stopwords from tokenized text.
        
        Parameters:
        - tokens: Tokenized text
        - custom_stopwords: Additional stopwords to remove
        
        Educational Notes:
        - Stopwords are common words that add little meaning to text
        - Removing stopwords can reduce noise and improve model performance
        - Some stopwords may be important for certain tasks (e.g., "not" in sentiment analysis)
        - Custom stopwords can be domain-specific
        """
        if custom_stopwords:
            stop_words = self.stop_words.union(set(custom_stopwords.split('\n')))
        else:
            stop_words = self.stop_words
        
        if isinstance(tokens, pd.Series):
            return tokens.apply(lambda x: [word for word in x if word not in stop_words])
        else:  # DataFrame
            tokens['words'] = tokens['words'].apply(lambda x: [word for word in x if word not in stop_words])
            return tokens
    
    def normalize_words(self, tokens, method="lemmatization"):
        """
        Normalizes words through lemmatization or stemming.
        
        Parameters:
        - tokens: Tokenized text
        - method: Normalization method ("lemmatization" or "stemming")
        
        Educational Notes:
        - Word normalization reduces words to their base form
        - Lemmatization uses dictionary lookups and considers word context
        - Stemming uses rules to chop off word endings
        - Lemmatization is more accurate but slower than stemming
        - The choice between methods depends on the task and performance requirements
        """
        if method == "lemmatization":
            if isinstance(tokens, pd.Series):
                return tokens.apply(lambda x: [self.lemmatizer.lemmatize(word) for word in x])
            else:  # DataFrame
                tokens['words'] = tokens['words'].apply(lambda x: [self.lemmatizer.lemmatize(word) for word in x])
                return tokens
        else:  # stemming
            if isinstance(tokens, pd.Series):
                return tokens.apply(lambda x: [self.stemmer.stem(word) for word in x])
            else:  # DataFrame
                tokens['words'] = tokens['words'].apply(lambda x: [self.stemmer.stem(word) for word in x])
                return tokens

class SentimentAnalyzer:
    """
    A sentiment analysis class using pre-trained transformer models.
    
    This class uses the Hugging Face Transformers library to perform sentiment analysis.
    It can analyze text and determine whether it expresses positive or negative sentiment.
    
    Educational Notes:
    - Sentiment analysis is a common NLP task that determines the emotional tone of text
    - Transformer models like BERT and DistilBERT are state-of-the-art for sentiment analysis
    - The model outputs both a sentiment label and a confidence score
    - Sentiment analysis can be used for various applications like customer feedback analysis
    """
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initializes the sentiment analyzer with a pre-trained model.
        
        Parameters:
        - model_name: Name of the pre-trained model to use
        
        Educational Notes:
        - Different models may perform better for different types of text
        - Larger models may be more accurate but require more computational resources
        - The choice of model depends on the specific use case and available resources
        """
        self.analyzer = pipeline("sentiment-analysis", model=model_name)
    
    def analyze(self, texts, chunk_size=500):
        """
        Analyzes sentiment in a batch of texts.
        
        Parameters:
        - texts: List of texts to analyze
        - chunk_size: Number of texts to process at once
        
        Educational Notes:
        - Processing texts in chunks helps manage memory usage
        - The chunk size can be adjusted based on available resources
        - Error handling is important as some texts may be too long or contain unsupported characters
        """
        sentiments = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            try:
                result = self.analyzer(chunk.tolist())
                sentiments.extend(result)
            except Exception as e:
                print(f"Could not analyze some text chunks: {str(e)}")
                continue
        return pd.DataFrame(sentiments)

class WordFrequencyAnalyzer:
    """
    A class for analyzing word frequencies in text.
    
    This class provides tools for:
    1. Counting word frequencies
    2. Visualizing word distributions
    3. Generating word clouds
    
    Educational Notes:
    - Word frequency analysis helps understand the most important terms in a text
    - Word clouds provide a visual representation of word importance
    - Frequency analysis can be used for tasks like keyword extraction and topic modeling
    - Stopwords should typically be removed before frequency analysis
    """
    def __init__(self):
        pass
    
    def analyze(self, tokens):
        """
        Analyzes word frequencies in tokenized text.
        
        Parameters:
        - tokens: Tokenized text
        
        Educational Notes:
        - Word frequency is a basic but powerful text analysis technique
        - Common words may indicate important topics or themes
        - Rare words may be important for specific domains
        - Frequency analysis can be combined with other techniques for deeper insights
        """
        all_words = []
        if isinstance(tokens, pd.Series):
            for doc in tokens:
                all_words.extend(doc)
        else:  # DataFrame
            for doc in tokens['words']:
                all_words.extend(doc)
        
        word_freq = Counter(all_words)
        return pd.DataFrame(word_freq.most_common(), columns=['Word', 'Frequency'])
    
    def plot_word_frequency(self, word_freq, top_n=20):
        """
        Creates a bar plot of word frequencies.
        
        Parameters:
        - word_freq: DataFrame of word frequencies
        - top_n: Number of top words to display
        
        Educational Notes:
        - Bar plots help visualize the distribution of word frequencies
        - The top N words often represent the main topics or themes
        - The shape of the distribution can indicate the nature of the text
        - Long-tail distributions are common in natural language
        """
        top_words = word_freq.head(top_n)
        fig = px.bar(top_words, x='Word', y='Frequency', title='Word Frequency Distribution')
        return fig
    
    def generate_word_cloud(self, word_freq):
        """
        Generates a word cloud from word frequencies.
        
        Parameters:
        - word_freq: DataFrame of word frequencies
        
        Educational Notes:
        - Word clouds provide an intuitive visualization of word importance
        - Word size typically represents frequency or importance
        - Color can be used to represent additional dimensions
        - Word clouds are useful for quick text summarization
        """
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
            dict(word_freq.values)
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig

class TextClassifier(nn.Module):
    """
    A neural network model for text classification.
    
    This model uses:
    1. Word embeddings to represent text
    2. LSTM layers to process sequences
    3. Fully connected layers for classification
    
    Educational Notes:
    - Text classification is a fundamental NLP task
    - Word embeddings capture semantic meaning of words
    - LSTMs are effective for processing sequential data
    - The model architecture can be adapted for different classification tasks
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.5):
        """
        Initializes the text classifier model.
        
        Parameters:
        - vocab_size: Size of the vocabulary
        - embedding_dim: Dimension of word embeddings
        - hidden_dim: Dimension of LSTM hidden states
        - output_dim: Number of output classes
        - dropout: Dropout rate for regularization
        
        Educational Notes:
        - The embedding layer converts words to dense vectors
        - LSTM layers process the sequence of word embeddings
        - Dropout helps prevent overfitting
        - The output layer produces class probabilities
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, 
                           bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        """
        Forward pass of the model.
        
        Parameters:
        - text: Input text tensor
        
        Educational Notes:
        - The forward pass processes text through the network
        - Word embeddings capture semantic meaning
        - LSTM layers process the sequence
        - The final layer produces class predictions
        """
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

class TextGenerator(nn.Module):
    """
    A neural network model for text generation.
    
    This model uses:
    1. Word embeddings to represent text
    2. LSTM layers to generate sequences
    3. A fully connected layer to predict the next word
    
    Educational Notes:
    - Text generation is a challenging NLP task
    - The model learns to predict the next word in a sequence
    - Temperature can be used to control generation randomness
    - Different sampling strategies can be used for generation
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.5):
        """
        Initializes the text generator model.
        
        Parameters:
        - vocab_size: Size of the vocabulary
        - embedding_dim: Dimension of word embeddings
        - hidden_dim: Dimension of LSTM hidden states
        - num_layers: Number of LSTM layers
        - dropout: Dropout rate for regularization
        
        Educational Notes:
        - The model architecture is similar to the classifier but used differently
        - The output layer predicts the next word in the sequence
        - Multiple LSTM layers can capture more complex patterns
        - Dropout helps prevent overfitting during training
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                           dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, hidden=None):
        """
        Forward pass of the model.
        
        Parameters:
        - text: Input text tensor
        - hidden: Optional initial hidden state
        
        Educational Notes:
        - The model processes text one word at a time
        - The hidden state maintains information about the sequence
        - The output predicts the probability of each possible next word
        - Different sampling strategies can be used during generation
        """
        embedded = self.embedding(text)
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        return self.fc(output), hidden 