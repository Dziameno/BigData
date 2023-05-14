import nltk
import string
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt

file = open("hammerheadShark.txt", "r")
text = file.read()

tokens = word_tokenize(text)
print("Number of tokens: ", len(tokens))

stopwords = nltk.corpus.stopwords.words("english")
tokens = [token for token in tokens if token not in stopwords]
print("Number of tokens after removing stopwords: ", len(tokens))

tokens = [token for token in tokens if token not in string.punctuation]
print("Number of tokens after punctuation removal: ", len(tokens))

tokens = [token for token in tokens if token != '"' and token != "''"
          and token != "``" and token != "’" and token != "“" and token != "”"
          and token != "–" and token != "..." and token != "‘" and token != "—"
          and token != "..." and token != "–" and token != "..." and token != "–"]

wnl = nltk.WordNetLemmatizer()
tokens = [wnl.lemmatize(token) for token in tokens]
print("Number of tokens after lemmatization: ", len(tokens))


freq = nltk.FreqDist(tokens)
freq.plot(10, cumulative=False, title="10 Most Frequent Tokens")


wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("shark_wordcloud.png")