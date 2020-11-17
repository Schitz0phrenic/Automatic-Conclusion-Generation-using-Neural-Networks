from transformers import BartTokenizer, BartForConditionalGeneration


class BARTInferenceModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_text(self, text):
        inputs = self.tokenizer([text], return_tensors='pt')
        summary_ids = self.model.generate(inputs['input_ids'], num_beams=4, max_length=128, early_stopping=True)
        conclusion = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                      for g in summary_ids]
        return conclusion[0].strip()

    @staticmethod
    def load(path: str):
        model = BartForConditionalGeneration.from_pretrained(path)
        tokenizer = BartTokenizer.from_pretrained(path)
        return BARTInferenceModel(model, tokenizer)
