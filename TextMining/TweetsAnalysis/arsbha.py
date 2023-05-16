import nltk
import string
import csv
import text2emotion as te
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

open_file = open("ArsenalBrighton_croped.csv", "r")
csv_file = csv.reader(open_file)

tweets = []
locations = []
all_tokens = []
pos_tweets = []
neg_tweets = []

happy = []
angry = []
surprise = []
sad = []
fear = []

for row in csv_file:
    tweets.append([row[2], row[3]])

sid = SentimentIntensityAnalyzer()

for tweet in tweets:
    tokens = nltk.word_tokenize(tweet[0])
    all_tokens.extend(tokens)

    stopwords = nltk.corpus.stopwords.words("english")
    all_tokens = [token for token in all_tokens if token not in stopwords]

    all_tokens = [token for token in all_tokens if token not in string.punctuation]

    wnl = nltk.WordNetLemmatizer()
    all_tokens = [wnl.lemmatize(token) for token in all_tokens]

    all_tokens = [token for token in all_tokens if token.isalpha()]
    all_tokens = [token for token in all_tokens if len(token) > 2]
    all_tokens = [token for token in all_tokens if token != "http" and token != "https"
                  and token != "ARSBHA" and token != "Arsenal" and token != "Brighton" and token != "BHAFC"
                  and token != "amp" and token != "co" and token != "RT" and token != "arsenal" and token!= "bha"
                  and token != "afc" and token != "arsenal" and token != "brighton" and token != "arsenalbrighton"
                  and token != "PremierLeague" and token != "premierleague" and token != "PL" and token != "pl"]

    locations.append(tweet[1])

    score = sid.polarity_scores(" ".join(all_tokens))
    if score["compound"] >= 0.05:
        pos_tweets.append(" ".join(all_tokens))
    elif score["compound"] <= 0.05:
        neg_tweets.append(" ".join(all_tokens))

    emotions = te.get_emotion(tweet[0])

    if emotions["Happy"] > 0.6:
        happy.append(" ".join(all_tokens))
    if emotions["Angry"] > 0.2:
        angry.append(" ".join(all_tokens))
    if emotions["Surprise"] > 0.6:
        surprise.append(" ".join(all_tokens))
    if emotions["Sad"] > 0.2:
        sad.append(" ".join(all_tokens))
    if emotions["Fear"] > 0.2:
        fear.append(" ".join(all_tokens))





freq_dist = nltk.FreqDist(all_tokens)
freq_dist.plot(20, title="Top 20 words in #ARSBHA tweets")

# wordcloud = WordCloud().generate(" ".join(all_tokens))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()
# wordcloud.to_file("#ARSBHA_wordcloud.png")
#
# wordcloud = WordCloud().generate(" ".join(locations))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()
# wordcloud.to_file("#ARSBHA_locations_wordcloud.png")
#
# #create wordclouds for positive and negative tweets
# wordcloud = WordCloud().generate(" ".join(pos_tweets))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()
# wordcloud.to_file("#ARSBHA_positive_wordcloud.png")

# wordcloud = WordCloud().generate(" ".join(happy))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()
# wordcloud.to_file("#ARSBHA_happy_wordcloud.png")
#
# wordcloud = WordCloud().generate(" ".join(angry))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()
# wordcloud.to_file("#ARSBHA_angry_wordcloud.png")
#
# wordcloud = WordCloud().generate(" ".join(surprise))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()
# wordcloud.to_file("#ARSBHA_surprise_wordcloud.png")
#
# wordcloud = WordCloud().generate(" ".join(sad))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()
# wordcloud.to_file("#ARSBHA_sad_wordcloud.png")
#
# wordcloud = WordCloud().generate(" ".join(fear))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()
# wordcloud.to_file("#ARSBHA_fear_wordcloud.png")