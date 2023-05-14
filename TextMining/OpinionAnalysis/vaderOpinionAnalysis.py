from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import text2emotion as te

posOpinion = pd.read_csv("positive.txt")
negOpinion = pd.read_csv("negative.txt")
posEdited = pd.read_csv("positiveEdited.txt")
negEdited = pd.read_csv("negativeEdited.txt")

sid = SentimentIntensityAnalyzer()
opinions = [posOpinion, posEdited, negOpinion, negEdited]

for opinion in opinions:
    for sentence in opinion:
        score = sid.polarity_scores(sentence)
        print("Sentence was rated as ", score['pos'] * 100, "% Positive")
        print("Sentence was rated as ", score['neg'] * 100, "% Negative")
        print("Sentence was rated as ", score['neu'] * 100, "% Neutral")
        print("Compound Sentiment Score: ", score['compound'])
        print("Emotions: ", te.get_emotion(sentence) , "\n")