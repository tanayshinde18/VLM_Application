import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
class SentimentAnalyzer:
    def __init__(self,model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", device=None):
        # Load 3-class sentiment model
        self.device=0 if(device=="cuda" or (device is None and torch.cuda.is_available())) else -1
        self.analyzer = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=self.device
        )
    def analyze(self, text):
        """
        Returns sentiment label (positive, negative, neutral) and confidence score.
        """
        result = self.analyzer(text,truncation=True)[0]
        label = result["label"].lower()
        score =float(result["score"])
        return label, score
