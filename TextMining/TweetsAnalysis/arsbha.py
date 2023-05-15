import nltk
import string
import csv
from wordcloud import WordCloud
import matplotlib.pyplot as plt

open_file = open("ArsenalBrighton_croped.csv", "r")
csv_file = csv.reader(open_file)

tweets = []
locations = []
all_tokens = []
for row in csv_file:
    tweets.append([row[2], row[3]])

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
                  and token != "ARSBHA" and token != "Arsenal" and token != "Brighton"
                  and token != "amp" and token != "co" and token != "RT" and token != "arsenal"]

    locations.append(tweet[1])


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