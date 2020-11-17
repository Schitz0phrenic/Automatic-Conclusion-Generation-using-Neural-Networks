from TargetInference import load_fasttext, TargetInference
import torch
from os import environ

if torch.cuda.is_available():
    environ["CUDA_VISIBLE_DEVICES"] = "0"

# Premises
p1 = "Metal music turns people violent."
p2 = "There is no correlation between violent crimes and the consumption of metal music."
p3 = "People found out that working with fast paced music raises productivity."

# Load models
embedding = TargetInference.load_embedding_model("./models/TargetInference/embedding.pt",
                                                 "./models/TargetInference/targets")
ranking = TargetInference.load_ranked_model("./models/TargetInference/ranking.pickle")
tagger = TargetInference.load_target_extraction_model("./models/tagger2.pt")
target_inference = TargetInference(tagger, ranking, embedding)

# load fasttext
load_fasttext("./models/TargetInference/crawl.bin")

print(target_inference.infer_target([p1, p2, p3]))
