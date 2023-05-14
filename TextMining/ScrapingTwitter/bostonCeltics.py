#pip3 install --upgrade git+https://github.com/JustAnotherArchivist/snscrape.git

import snscrape.modules.twitter as snt
import pandas as pd
import datetime as dt
import os

## NOT WORKING BECAUSE OF THE NEW TWITTER API

# tweetNumber = 100
# tweet_list = []
#
# start_date = dt.date(2023, 4, 12)
# end_date = dt.date(2023, 5, 14)
#
# search = '#bleedgreen since:' + str(start_date) + ' until:' + str(end_date)
#
# for i, tweet in enumerate(snt.TwitterSearchScraper(search).get_items()):
#     if i > tweetNumber:
#         break
#     tweet_list.append([tweet.id, tweet.user.username, tweet.content, tweet.user.location])
#
# df = pd.DataFrame(tweet_list, columns=['id', 'username', 'content', 'location'])
# df.to_csv('bostonCeltics.csv', sep=',', index=False)

#USING OS TO SCRAPE WITH COMMAND LINE
#snscrape --jsonl --max-results 100 --since 2023-04-12 twitter-search "#bleedgreen until:2023-05-14" > bostonCeltics.json

DATE_START = input("Enter the start date in YYYY-MM-DD format i.e 2023-05-10: \n")
DATE_END = input("Enter the end date in YYYY-MM-DD format i.e 2023-05-10: \n")
HASHTAG = input("Enter the hashtag: \n")
FILENAME = input("Enter the output filename: \n")
MAX_LIMIT = input("Enter the max limit:\n")


def sns_scrape():
    os.system(
        f'snscrape --jsonl '
        f'--max-results {MAX_LIMIT} '
        f'--since {DATE_START} '
        f'twitter-search "{HASHTAG} '
        f'until:{DATE_END}" > {FILENAME}.json')


if __name__ == "__main__":
    sns_scrape()
    filePath = FILENAME + '.json'
    csvPath = FILENAME + '.csv'
    print(filePath)
    print(csvPath)
    print(FILENAME + '_croped.csv')
    df = pd.read_json(filePath, lines=True)
    df.to_csv(csvPath)
    df['place'] = df['place'].apply(lambda x: x['fullName'] if x != None else None)
    df = df[['url','date', 'rawContent', 'place', 'likeCount']]
    df = df.sort_values(by=['likeCount'], ascending=False)
    df.to_csv(FILENAME + '_croped.csv', index=False)
    exit()