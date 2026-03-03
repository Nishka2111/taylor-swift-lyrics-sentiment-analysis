# taylor-swift-lyrics-sentiment-analysis
Natural Language Processing project analyzing word frequency and sentiment patterns in Taylor Swift lyrics (2006–2020) using NLTK and VADER.

Project Overview
This project performs Natural Language Processing (NLP) analysis on a dataset of Taylor Swift song lyrics from 2006 to 2020.

The goal of this project is to:
- Clean and preprocess textual data
- Perform word frequency analysis
- Apply sentiment analysis using VADER
- Extract meaningful insights from lyrical patterns
-This project demonstrates practical implementation of NLP techniques using Python.

Dataset
The dataset contains song lyrics from multiple albums released between 2006 and 2020.The dataset includes:
-Song titles
-Album names
-Lyrics text

Technologies Used
-Python
-Pandas
-NLTK
-Matplotlib
-Seaborn

Key Features
1. Text Preprocessing
-Lowercasing
-Removal of punctuation
-Tokenization using NLTK
-Stopword removal

2. Word Frequency Analysis
-Token-based word counting
-Identification of most common words
-Frequency ranking

3. Sentiment Analysis
-VADER SentimentIntensityAnalyzer
-Compound sentiment scoring
-Classification into positive, negative, and neutral sentiments

How to Run This Project
Clone the repository:
git clone https://github.com/your-username/taylor-swift-lyrics-sentiment-analysis.git
Install dependencies:
pip install pandas nltk matplotlib seaborn
Download required NLTK resources:

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('vader_lexicon')
nltk.download('stopwords')

Run the script:
python sentimentanalysis.py

Sample Output
Top 20 most frequent words in the dataset
Sentiment scores for each song
Distribution of sentiment categories

Learning Outcomes
-Practical implementation of NLP preprocessing techniques
-Working with real-world textual datasets
-Applying lexicon-based sentiment analysis
-Debugging dependency and environment issues in Python
