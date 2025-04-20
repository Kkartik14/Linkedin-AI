import pandas as pd
import nltk
from textblob import TextBlob
from datetime import datetime, timedelta
import re
import os
import logging
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(nltk.corpus.stopwords.words("english"))

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def load_posts(filename="data/posts.csv"):
    """Load posts from CSV."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found. Run scraper first.")
    df = pd.read_csv(filename)
    logger.info(f"Loaded {len(df)} posts from {filename}")
    return df

def preprocess_data(df):
    """Clean and preprocess the post data."""
    logger.info("Preprocessing data")

    df = df.drop_duplicates(subset=["profile", "text", "date"])
    logger.info(f"Removed duplicates, {len(df)} posts remain")

    def clean_hashtags(hashtags_str):
        if pd.isna(hashtags_str):
            return []
        try:
            hashtags = eval(hashtags_str)
            return [re.sub(r"hashtag$", "", tag, flags=re.IGNORECASE) for tag in hashtags]
        except:
            return []
    df["hashtags"] = df["hashtags"].apply(clean_hashtags)

    def parse_date(date_str):
        if pd.isna(date_str):
            return None, None
        try:
            if "ago" in date_str.lower():
                match = re.search(r"(\d+)([dhwmo])", date_str.lower())
                if match:
                    value, unit = int(match.group(1)), match.group(2)
                    if unit == "d":
                        delta = timedelta(days=value)
                    elif unit == "h":
                        delta = timedelta(hours=value)
                    elif unit == "w":
                        delta = timedelta(weeks=value)
                    elif unit == "mo":
                        delta = timedelta(days=value * 30)
                    else:
                        delta = timedelta(0)
                    post_time = datetime.now() - delta
                    return post_time.hour, post_time.strftime("%A")
            return pd.to_datetime(date_str).hour, pd.to_datetime(date_str).strftime("%A")
        except:
            return None, None
    df[["post_hour", "post_day"]] = df["date"].apply(parse_date).apply(pd.Series)

    for col in ["likes", "comments", "shares"]:
        df[col] = df[col].fillna(0).astype(int)

    df["text"] = df["text"].str.strip().str.replace(r"\s+", " ", regex=True)

    return df

def get_bert_embeddings(texts):
    """Generate BERT embeddings for a list of texts."""
    logger.info("Generating BERT embeddings")
    embeddings = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            embeddings.append(np.zeros(384))  
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)
    return np.array(embeddings)

def cluster_posts(embeddings, n_clusters=5):
    """Cluster posts using K-means on BERT embeddings."""
    logger.info(f"Clustering posts into {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels

def extract_features(df):
    """Extract features from posts, including BERT-based clustering and sentiment."""
    logger.info("Extracting features")
    df["length"] = df["text"].apply(len)
    df["num_hashtags"] = df["hashtags"].apply(len)

    def get_sentiment(text):
        if not isinstance(text, str) or not text.strip():
            return 0.0, "neutral"
        result = sentiment_analyzer(text[:512])[0]  
        score = result["score"] if result["label"] == "POSITIVE" else -result["score"]
        tone = "positive" if score > 0.3 else "negative" if score < -0.3 else "neutral"
        return score, tone
    df[["sentiment", "tone"]] = df["text"].apply(get_sentiment).apply(pd.Series)
    cta_patterns = r"(learn more|comment below|check out|join us|click here|dm me|rsvp|apply now)"
    df["has_cta"] = df["text"].str.contains(cta_patterns, case=False, na=False)
    embeddings = get_bert_embeddings(df["te" \
    "xt"].tolist())
    df["topic_cluster"] = cluster_posts(embeddings, n_clusters=5)
    def get_cluster_keywords(cluster_id):
        cluster_texts = df[df["topic_cluster"] == cluster_id]["text"].str.lower()
        words = [word for text in cluster_texts for word in nltk.word_tokenize(text) if word.isalnum() and word not in stop_words]
        return [word for word, _ in Counter(words).most_common(5)]
    topic_labels = {i: ", ".join(get_cluster_keywords(i)) for i in range(5)}
    df["topic_label"] = df["topic_cluster"].map(topic_labels)

    return df

def analyze_engagement(df):
    """Analyze engagement metrics and trends."""
    logger.info("Analyzing engagement")
    df["total_engagement"] = df["likes"] + df["comments"] + df["shares"]
    trends = {
        "avg_engagement_by_hour": df.groupby("post_hour")["total_engagement"].mean().round(2).to_dict(),
        "avg_engagement_by_day": df.groupby("post_day")["total_engagement"].mean().round(2).to_dict(),
        "avg_engagement_by_tone": df.groupby("tone")["total_engagement"].mean().round(2).to_dict(),
        "avg_engagement_by_has_cta": df.groupby("has_cta")["total_engagement"].mean().round(2).to_dict(),
        "avg_engagement_by_length": df.groupby(
            pd.cut(df["length"], bins=[0, 100, 500, 1000, float("inf")], labels=["short", "medium", "long", "very long"])
        )["total_engagement"].mean().round(2).to_dict(),
        "top_hashtags": pd.Series([tag for tags in df["hashtags"] for tag in tags]).value_counts().head(5).to_dict(),
        "avg_engagement_by_topic": df.groupby("topic_label")["total_engagement"].mean().round(2).to_dict(),
        "avg_engagement_by_profile": df.groupby("profile")["total_engagement"].mean().round(2).to_dict(),
        "archit_top_hashtags": pd.Series([tag for tags in df[df["profile"] == "https://www.linkedin.com/in/archit-anand/"]["hashtags"] for tag in tags]).value_counts().head(5).to_dict(),
        "archit_top_topics": df[df["profile"] == "https://www.linkedin.com/in/archit-anand/"].groupby("topic_label")["total_engagement"].mean().round(2).to_dict(),
    }

    return trends, df

def save_outputs(df, trends, posts_filename="data/posts_analyzed.csv", trends_filename="data/trends.csv"):
    """Save enriched posts and trends to CSV."""
    os.makedirs("data", exist_ok=True)
    df.to_csv(posts_filename, index=False)
    logger.info(f"Saved enriched posts to {posts_filename}")
    trends_df = pd.DataFrame([{"metric": k, "value": v} for k, v in trends.items()])
    trends_df.to_csv(trends_filename, index=False)
    logger.info(f"Saved trends to {trends_filename}")

def main():
    df = load_posts()
    df = preprocess_data(df)
    df = extract_features(df)
    trends, df = analyze_engagement(df)
    logger.info("Key Trends:")
    for metric, value in trends.items():
        logger.info(f"{metric}: {value}")
    save_outputs(df, trends)

if __name__ == "__main__":
    main()