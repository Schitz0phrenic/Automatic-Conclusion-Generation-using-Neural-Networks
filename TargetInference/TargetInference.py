from .conclusion_target_modeling import TargetSpaceModeling, RankedTargetModel
from .AlshomaryHybrid import HybridExtractor
from TargetIdentification import TargetIdentifier
from typing import List


class TargetInference:
    def __init__(self, target_tagger: TargetIdentifier,
                 ranked_model: RankedTargetModel, embedded_model: TargetSpaceModeling):
        self._model_ = HybridExtractor(embedded_model, ranked_model)
        self._identifier_ = target_tagger

    def infer_target(self, premises: List[str]):
        return self._model_.get_target(premises, self._identifier_)

    @staticmethod
    def load_ranked_model(path: str):
        return RankedTargetModel(path)

    @staticmethod
    def load_embedding_model(path: str, knowledge_base_path: str):
        return TargetSpaceModeling(path, load_concepts=False, src_target_space_path=knowledge_base_path)

    @staticmethod
    def load_target_extraction_model(path):
        return TargetIdentifier.load(path)