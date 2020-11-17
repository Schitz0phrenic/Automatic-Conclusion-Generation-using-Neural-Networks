from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import List
from TargetIdentification import TargetIdentifier


def nltk_sentiment(premises: List[str]):
    sentence = " ".join(premises)
    sentiment = SentimentIntensityAnalyzer()
    score = sentiment.polarity_scores(sentence)
    # sentiment = {"text": sentence, "sentiment": score["compound"]}
    sentiment = score["compound"]
    return sentiment


def get_np(premises: List[str]):
    text = " ".join(premises)
    blob = TextBlob(text)
    return list(blob.noun_phrases)


def get_target_noun_phrases(premises: List[str], target_identifier: TargetIdentifier):
    premise_targets = [target[0] for p in premises for target in target_identifier.predict(p)]
    return set(get_np(premise_targets))


def get_text_noun_phrases(premises: List[str]):
    return get_np(premises)
