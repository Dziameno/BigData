# pip3 install --upgrade git+https://github.com/JustAnotherArchivist/snscrape.git

import pandas as pd
import datetime as dt
import os
import json

DATE_START = input("Enter the start date in YYYY-MM-DD format: \n")
DATE_END = input("Enter the end date in YYYY-MM-DD: \n")
HASHTAG = input("Enter the hashtag without #: \n")
FILENAME = input("Enter the output filename: \n")
MAX_LIMIT = input("Enter the max limit for each day:\n")


def iterate_over_dates(date_start, date_end):
    date_start = dt.datetime.strptime(date_start, '%Y-%m-%d')
    date_end = dt.datetime.strptime(date_end, '%Y-%m-%d')
    date_end += dt.timedelta(days=1)
    date_generated = [date_start + dt.timedelta(days=x) for x in range(0, (date_end - date_start).days)]
    return date_generated


def run_sns_scrape(date_start, date_end, hashtag, filename, max_limit):
    command = f'snscrape --jsonl --max-results {max_limit} --since {date_start} ' \
              f'twitter-search "{hashtag} until:{date_end}" >> {filename}.json'
    os.system(command)


def remove_duplicates_json(filename):
    with open(filename, 'r') as json_file:
        tweets = []
        for line in json_file:
            tweet = json.loads(line)
            if tweet['id'] not in {t['id'] for t in tweets}:
                tweets.append(tweet)

    with open(filename, 'w') as json_file:
        for tweet in tweets:
            json.dump(tweet, json_file)
            json_file.write('\n')


def convert_json_to_csv(json_file, csv_file):
    tweets = []
    with open(json_file, 'r') as f:
        for line in f:
            tweet = json.loads(line)
            tweets.append(tweet)

    df = pd.json_normalize(tweets)
    df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    date_generated = iterate_over_dates(DATE_START, DATE_END)
    for date in date_generated:
        run_sns_scrape(date.strftime("%Y-%m-%d"),
                       (date + dt.timedelta(days=1)).strftime("%Y-%m-%d"),
                       HASHTAG,
                       FILENAME,
                       MAX_LIMIT)
        remove_duplicates_json(FILENAME + ".json")

    json_filename = FILENAME + ".json"
    csv_filename = FILENAME + ".csv"
    convert_json_to_csv(json_filename, csv_filename)

    exit()