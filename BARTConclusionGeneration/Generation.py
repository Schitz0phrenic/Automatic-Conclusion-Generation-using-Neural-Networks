from typing import Optional, Union
from BARTConclusionGeneration.BARTModel import BARTInferenceModel
from TargetIdentification import TargetIdentifier


class BARTGenerator:
    def __init__(self,
                 bart_model: [str, BARTInferenceModel],
                 target_identifier: Optional[Union[str, TargetIdentifier]] = None):
        if isinstance(bart_model, str):
            self._model_ = BARTInferenceModel.load(bart_model)
        elif isinstance(bart_model, BARTInferenceModel):
            self._model_ = bart_model
        else:
            raise ValueError("The parameter 'bart_model' must be the path to a BART model or a BARTModel object")
        self._tagger_ = None
        if target_identifier is not None:
            if isinstance(target_identifier, str):
                self._tagger_ = TargetIdentifier.load(target_identifier)
            elif isinstance(target_identifier, TargetIdentifier):
                self._tagger_ = target_identifier
            else:
                raise ValueError("The parameter 'target_identifier' must be the path to the corresponding model "
                                 "or a TargetIdentifier object")

    def generate_conclusion(self, text: str):
        if self._tagger_ is not None:
            targets = self._tagger_.predict(text)
            for t in targets:
                target = t[0].strip()
                if target in text:
                    text = text.replace(target, f"<T-B> {target} <T-E>")
                    break
        return self._model_.generate_text(text).replace("<T-B> ", "").replace(" <T-E>", "").strip()
