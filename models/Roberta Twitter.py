# %%
#this needsto be updated
token=""

# %%
import tweepy
import pandas as pd
from datetime import datetime, timedelta
from transformers import pipeline
from time import sleep
import os
from pandas.errors import EmptyDataError

# -------------- CONFIGURATION --------------
BEARER_TOKEN = token
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

# --- Authenticate & Load Model ---
client = tweepy.Client(bearer_token=BEARER_TOKEN)

sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    framework="pt",
    return_all_scores=True
)

def classify_continuous(text):
    try:
        scores = sentiment_model(text[:512])[0]
        label_weights = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
        return sum(label_weights[item['label']] * item['score'] for item in scores)
    except:
        return 0

# --- Load existing CSV if available ---
if os.path.exists(CSV_PATH):
    try:
        existing_df = pd.read_csv(CSV_PATH, parse_dates=["date"])
        existing_dates = set(existing_df["date"].dt.date)
        print(f"Loaded existing sentiment file with {len(existing_df)} entries.")
    except EmptyDataError:
        print(" File exists but is empty. Starting fresh.")
        existing_df = pd.DataFrame(columns=["date", "score"])
        existing_dates = set()
else:
    print("No existing sentiment file found. Starting fresh.")
    existing_df = pd.DataFrame(columns=["date", "score"])
    existing_dates = set()

# --- Determine collection range ---
today = datetime.utcnow().date()
start_date = API_ACTIVATION_DATETIME.date()
end_date = today - timedelta(days=1)

print(f"\nCollecting sentiment from {start_date} to {end_date}...")

# --- Sentiment loop ---
all_data = []

for i in range((end_date - start_date).days + 1):
    day = start_date + timedelta(days=i)

    if day in existing_dates:
        print(f"Skipping {day} (already collected)")
        continue

    print(f"\nFetching tweets for: {day}")

    # Handle partial first day
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

# --- Save updated CSV ---
new_df = pd.DataFrame(all_data)
combined_df = pd.concat([existing_df, new_df])
combined_df = combined_df.drop_duplicates(subset="date").sort_values("date")
combined_df.to_csv(CSV_PATH, index=False)

print(f"\nUpdated sentiment data saved to: {CSV_PATH}")


# %%
pip install tweepy pandas transformers


