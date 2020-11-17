from flair.data import Sentence
from TargetIdentification import TargetIdentifier
from typing import List
from nltk.corpus import stopwords


STOPWORDS = None


def load_stopwords():
    global STOPWORDS
    STOPWORDS = stopwords.words("english") + [".", ",", "!", "?", ":", ";"]


class Argument:
    def __init__(self, *premises):
        self.texts = premises
        self.claims = []
        self.target = None
        self.started = False

    def to_dict(self):
        return {"claims": self.claims, "target": self.target}

    def start(self, identifier: TargetIdentifier):
        for p in self.texts:
            parsed = {"text": p, "targets": [{"text": target} for target, _ in identifier.predict(p)]}
            self.claims.append(parsed)
        self.started = True

    def get_target(self, model):
        self.target = model.get_target(self.claims)
        return self.target


class HybridExtractor:
    def __init__(self, preferred_model, fallback_model):
        self.m1 = preferred_model
        self.m2 = fallback_model

    def get_target(self, argument: List[str], tagger: TargetIdentifier = None):
        if isinstance(argument, (list, tuple)):
            argument = Argument(*argument)
            if not tagger:
                raise ValueError("You have to define a tagger if you try to extract from strings.")
            argument.start(tagger)
        target = argument.get_target(self.m1)
        overlapping = False
        for premise in argument.texts:
            if HybridExtractor.is_overlapping(target, premise):
                overlapping = True
                break
        if overlapping:
            return target
        target = argument.get_target(self.m2)
        return target

    @staticmethod
    def is_overlapping(text1: str, text2: str):
        global STOPWORDS
        if STOPWORDS is None:
            load_stopwords()
            assert STOPWORDS is not None
        tokens1 = HybridExtractor.get_token_set(text1.lower())
        tokens2 = HybridExtractor.get_token_set(text2.lower())
        t1 = set([t for t in tokens1 if t not in STOPWORDS])
        t2 = set([t for t in tokens2 if t not in STOPWORDS])
        return len(t1.intersection(t2)) > 0

    @staticmethod
    def get_token_set(text):
        return set([t.text for t in Sentence(text).tokens])
