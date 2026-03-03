import pandas as pd
import matplotlib.pyplot as plt
import collections
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer


# -----------------------------
# Configuration
# -----------------------------

ALBUM_YEAR_MAP = {
    "Taylor Swift": 2006,
    "Fearless (Taylor’s Version)": 2008,
    "Speak Now (Deluxe)": 2010,
    "Red (Deluxe Edition)": 2012,
    "1989 (Deluxe)": 2014,
    "reputation": 2017,
    "Lover": 2019,
    "folklore (deluxe version)": 2020,
    "evermore (deluxe version)": 2020,
}

NIGHT_WORDS = ['night', 'midnight', 'dawn', 'dusk', 'evening', 'late', 'dark']
DAY_WORDS = ['day', 'morning', 'light', 'sun', 'noon', 'golden', 'bright']
TIME_WORDS = ['today', 'tomorrow', 'yesterday']

STOPWORDS = ['the', 'a', 'this', 'that', 'to', 'is', 'am', 'was', 'were',
             'be', 'being', 'been']


# -----------------------------
# Data Loading
# -----------------------------

def load_data(filepath):
    df = pd.read_csv(filepath)
    df["album_year"] = df["album_name"].map(ALBUM_YEAR_MAP)
    return df


# -----------------------------
# Text Cleaning
# -----------------------------

def clean_text(df):
    df["clean_lyric"] = df["lyric"].str.lower()
    df["clean_lyric"] = df["clean_lyric"].str.replace(r'[^\w\s]', '', regex=True)
    df["clean_lyric"] = df["clean_lyric"].apply(
        lambda x: ' '.join([word for word in x.split() if word not in STOPWORDS])
    )
    return df


# -----------------------------
# Thematic Analysis
# -----------------------------

def add_theme_flags(df):
    night_regex = '|'.join(NIGHT_WORDS)
    day_regex = '|'.join(DAY_WORDS)
    time_regex = '|'.join(TIME_WORDS)

    df["night_flag"] = df["clean_lyric"].str.contains(night_regex)
    df["day_flag"] = df["clean_lyric"].str.contains(day_regex)
    df["time_flag"] = df["clean_lyric"].str.contains(time_regex)

    return df


def plot_theme_trends(df):
    yearly = df.groupby("album_year")[["night_flag", "day_flag"]].sum().reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(yearly["album_year"], yearly["night_flag"], label="Night References")
    plt.plot(yearly["album_year"], yearly["day_flag"], label="Day References")
    plt.xlabel("Album Year")
    plt.ylabel("Number of Mentions")
    plt.title("Night vs Day References Over Time")
    plt.legend()
    plt.show()


# -----------------------------
# Word Frequency Analysis
# -----------------------------

def word_frequency_analysis(df):
    df["tokens"] = df["clean_lyric"].apply(word_tokenize)
    word_list = [word for row in df["tokens"] for word in row]

    frequency = collections.Counter(word_list)
    most_common = frequency.most_common(20)

    print("\nTop 20 Most Common Words:")
    for word, count in most_common:
        print(f"{word}: {count}")


# -----------------------------
# Sentiment Analysis
# -----------------------------

def sentiment_analysis(df):
    sia = SentimentIntensityAnalyzer()

    df["sentiment"] = df["clean_lyric"].apply(lambda x: sia.polarity_scores(x))
    df[["neg", "neu", "pos", "compound"]] = df["sentiment"].apply(pd.Series)

    yearly_sentiment = df.groupby("album_year")["compound"].sum().reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(yearly_sentiment["album_year"], yearly_sentiment["compound"])
    plt.xlabel("Album Year")
    plt.ylabel("Compound Sentiment Score")
    plt.title("Sentiment Trend Over Time")
    plt.show()

    return df


def compare_day_night_sentiment(df):
    night_df = df[df["night_flag"]]
    day_df = df[df["day_flag"]]

    print("\nDay vs Night Sentiment Comparison:")
    print("Night sentiment:", night_df["compound"].sum())
    print("Day sentiment:", day_df["compound"].sum())


# -----------------------------
# Main Execution
# -----------------------------

def main():
    filepath = "taylor_swift_lyrics_2006-2020_all.csv"

    df = load_data(filepath)
    df = clean_text(df)
    df = add_theme_flags(df)

    plot_theme_trends(df)
    word_frequency_analysis(df)

    df = sentiment_analysis(df)
    compare_day_night_sentiment(df)


if __name__ == "__main__":
    main()