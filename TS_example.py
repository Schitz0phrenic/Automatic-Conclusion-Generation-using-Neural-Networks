from TargetStanceConclusionGeneration import TSPipeline, get_text_noun_phrases, nltk_sentiment
from TargetInference import TargetInference, load_fasttext


# load fasttext
load_fasttext("./models/TargetInference/crawl.bin")


# Load models
embedding = TargetInference.load_embedding_model("./models/TargetInference/embedding.pt",
                                                 "./models/TargetInference/targets")
ranking = TargetInference.load_ranked_model("./models/TargetInference/ranking.pickle")
tagger = TargetInference.load_target_extraction_model("./models/tagger2.pt")
target_inference = TargetInference(tagger, ranking, embedding)

ts_approach = TSPipeline("models/GPT", nltk_sentiment, get_text_noun_phrases, target_inference,
                         "models/pplm_settings.ini", "pplm-params")

# Definition of the argument
p1 = "Metal music turns people violent."
p2 = "There is no correlation between violent crimes and the consumption of metal music."
p3 = "People found out that working with fast paced music raises productivity."

conclusion = ts_approach.generate_conclusion([p1, p2, p3], False)
print(conclusion)
