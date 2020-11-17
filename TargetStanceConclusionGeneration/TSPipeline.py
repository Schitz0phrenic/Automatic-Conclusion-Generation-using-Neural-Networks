from .language_generation import gen
from typing import Callable, List, Set
from TargetInference import TargetInference
import sys

STDOUT = sys.stdout


class DummyFile:
    def write(self, x): pass


class TSPipeline:
    def __init__(self, gpt_path: str, stance_extractor: Callable[[List[str]], float],
                 noun_phrase_extractor: Callable[[List[str]], Set[str]],
                 target_inference: TargetInference,
                 pplm_config_path: str,
                 pplm_config_key: str):
        self.se = stance_extractor
        self.npe = noun_phrase_extractor
        self.ti = target_inference
        self.conf_path = pplm_config_path
        self.conf_key = pplm_config_key
        self.gpt = gpt_path

    def generate_conclusion(self, premises: List[str], suppress_outputs=False):
        if suppress_outputs:
            sys.stdout = DummyFile()
        try:
            stance = self.se(premises)
            bow = self.npe(premises)
            target = self.ti.infer_target(premises)
            # try:
            conclusion = gen(self.gpt, self.conf_path, self.conf_key, target, list(bow), stance)
            # except Exception as e:
            #   conclusion = "<UNKNOWN>"
        finally:
            sys.stdout = STDOUT
        return conclusion

