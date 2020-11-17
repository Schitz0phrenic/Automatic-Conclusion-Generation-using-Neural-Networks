from flair.models import SequenceTagger
from flair.data import Sentence
from typing import Union


class TargetIdentifier:
    def __init__(self, model: SequenceTagger):
        self.model = model

    @classmethod
    def load(cls, model_dir: str):
        tagger = SequenceTagger.load(model_dir)
        return TargetIdentifier(tagger)

    def predict(self, text: Union[str, Sentence]):
        if isinstance(text, str):
            text = Sentence(text)
        elif not isinstance(text, Sentence):
            raise ValueError("Targets can only be identified from str or Sentence objects.")
        self.model.predict(text)
        return TargetIdentifier.__sample_tagged__(text)

    @staticmethod
    def __sample_tagged__(sentence: Sentence):
        tokens = list(filter(lambda x: "ct" in x.get_tag("ct").value.lower(), sentence.tokens))
        if len(tokens) == 0:
            return []
        targets = []
        current = [tokens.pop(0)]
        for token in tokens:
            if "b-ct" in token.get_tag("ct").value.lower():
                text = " ".join([t.text for t in current])
                confidence = [t.get_tag("ct").score for t in current]
                confidence = sum(confidence) / len(confidence)
                targets.append((text, confidence))
                current = [token]
            else:
                current.append(token)
        text = " ".join([t.text for t in current])
        confidence = [t.get_tag("ct").score for t in current]
        confidence = sum(confidence) / len(confidence)
        targets.append((text, confidence))
        targets.sort(key=lambda x: x[1], reverse=True)
        return targets
