import torch
from transformers import pipeline


class SentimentAnalyzer:
    def __init__(
        self,
        model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=None,
    ):
        self.device = 0 if (device == "cuda" or (device is None and torch.cuda.is_available())) else -1
        self.analyzer = None

        try:
            self.analyzer = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
                device=self.device,
            )
        except Exception:
            self.analyzer = None

        self.negative_keywords = {
            "accident", "attack", "blood", "broken", "burning", "crash",
            "crime", "danger", "dead", "death", "destroyed", "explosion",
            "fight", "fire", "harm", "hit", "injured", "injury", "knife",
            "panic", "robbery", "scream", "smoke", "stolen", "suspicious",
            "theft", "unsafe", "violence", "weapon",
        }
        self.positive_keywords = {
            "calm", "clear", "fine", "normal", "peaceful", "safe", "smile",
            "stable", "okay",
        }

    def analyze(self, text):
        """
        Returns sentiment label (positive, negative, neutral) and confidence score.
        """
        if self.analyzer is not None:
            result = self.analyzer(text, truncation=True)[0]
            label = result["label"].lower()
            score = float(result["score"])
            return label, score

        return self._analyze_with_rules(text)

    def _analyze_with_rules(self, text):
        normalized_words = set(str(text).lower().split())
        negative_hits = len(self.negative_keywords & normalized_words)
        positive_hits = len(self.positive_keywords & normalized_words)

        if negative_hits > positive_hits and negative_hits > 0:
            return "negative", min(0.99, 0.55 + (0.08 * negative_hits))
        if positive_hits > negative_hits and positive_hits > 0:
            return "positive", min(0.99, 0.55 + (0.08 * positive_hits))
        return "neutral", 0.5
