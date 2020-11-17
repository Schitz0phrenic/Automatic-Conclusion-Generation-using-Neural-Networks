from BARTConclusionGeneration import BARTGenerator, BARTInferenceModel
from TargetIdentification import TargetIdentifier
import torch
from os import environ

if torch.cuda.is_available():
    environ["CUDA_VISIBLE_DEVICES"] = "0"

mode = "tagged"
if mode == "tagged":
    print("Using the tagged version!")
    target_identifier = TargetIdentifier.load("./models/tagger2.pt")
    bart = BARTInferenceModel.load("./models/BART-2/")
else:
    print("Using the untagged version")
    bart = BARTInferenceModel.load("./models/BART-0/")
    target_identifier = None


argument = "People who get good grades in school often do this by learning but not understanding. Since those " \
           "people tend do become lawyers, doctors and government officials the most important roles in " \
           "society are filled with people who can not think on their own but replicate what others say. " \
           "Hence, they often do things they do not understand."

generator = BARTGenerator(bart, target_identifier)
print(generator.generate_conclusion(argument))
