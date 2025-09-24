# %%
token=""

# %%
import tweepy
import pandas as pd
from datetime import datetime, timedelta
from transformers import pipeline
from time import sleep
import os

# -------------- CONFIGURATION --------------
BEARER_TOKEN =token
QUERY = (
    "bitcoin OR btc OR \"bitcoin price\" OR \"btc pump\" OR \"btc dump\" "
    "OR \"bitcoin crash\" OR \"bitcoin bull run\" OR \"bitcoin bear market\" "
    "OR \"bitcoin halving\" OR \"bitcoin ETF\" lang:en -is:retweet"
)
MAX_TWEETS_PER_DAY = 100
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))  # go up one level
CSV_PATH = os.path.join(ROOT_DIR, "NLP sentiment", "daily_sentiment.csv")
API_ACTIVATION_DATETIME = datetime(2025, 4, 3, 10, 23)  # UTC
# ------------------------------------------

# Auth and model
client = tweepy.Client(bearer_token=BEARER_TOKEN)
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    framework="pt",
    return_all_scores=True
)

# Continuous sentiment classifier
def classify_continuous(text):
    try:
        scores = sentiment_model(text[:512])[0]
        label_weights = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
        return sum(label_weights[item['label']] * item['score'] for item in scores)
    except:
        return 0

# Get valid date range
today = datetime.utcnow().date()
start_date = API_ACTIVATION_DATETIME.date()
end_date = today - timedelta(days=1)

print(f"Fetching from {start_date} to {end_date}...")

all_data = []

for i in range((end_date - start_date).days + 1):
    day = start_date + timedelta(days=i)
    print(f"\nFetching tweets for: {day}")

    # Handle partial first day (start after 10:23 UTC)
    if day == API_ACTIVATION_DATETIME.date():
        start_time = API_ACTIVATION_DATETIME
    else:
        start_time = datetime.combine(day, datetime.min.time())

    end_time = datetime.combine(day + timedelta(days=1), datetime.min.time())
    start_time_str = start_time.isoformat("T") + "Z"
    end_time_str = end_time.isoformat("T") + "Z"

    try:
        tweets = tweepy.Paginator(
            client.search_recent_tweets,
            query=QUERY,
            start_time=start_time_str,
            end_time=end_time_str,
            max_results=100,
            tweet_fields=["created_at", "text"]
        ).flatten(limit=MAX_TWEETS_PER_DAY)

        tweet_texts = [tweet.text for tweet in tweets]
        if not tweet_texts:
            print("No tweets found.")
            continue

        scores = [classify_continuous(text) for text in tweet_texts]
        avg_score = sum(scores) / len(scores)
        all_data.append({"date": day, "score": avg_score})
        print(f"Avg score: {avg_score:.3f} from {len(scores)} tweets")

    except tweepy.TooManyRequests:
        print("Rate limit hit — pausing for 60 seconds")
        sleep(60)
        continue
    except Exception as e:
        print(f"Error on {day}: {e}")
        continue

    sleep(10)  # Avoid rate limits

# Save to CSV
df = pd.DataFrame(all_data)
df.to_csv(CSV_PATH, index=False)
print(f"\nSaved sentiment data to {CSV_PATH}")


# %%



