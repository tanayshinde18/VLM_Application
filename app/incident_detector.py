import re
from typing import Dict, List


class IncidentDetector:
    """
    Rule-based NLP Incident Detector.
    Classifies a scene as SAFE, SUSPICIOUS, or DANGEROUS based on caption text.
    """

    def __init__(self):
        # --- Keyword Dictionaries ---

        # High-risk indicators (strong signals)
        self.high_risk_keywords = {
             # Accidents / Collisions
            "accident", "crash","crashed", "collision", "collided", "hit", "impact",
            "overturned", "rollover", "pileup", "wreck", "smashed",
            "rammed", "rearended", "skidded",

            # Fire / Explosion
            "fire", "smoke", "flames", "burning", "blaze",
            "explosion", "exploded", "blast", "sparks",
            "shortcircuit", "gasleak",

            # Violence / Physical Harm
            "fight", "fighting", "violence", "violent",
            "assault", "attacked", "beating", "beaten",
            "stabbing", "stabbed", "punching", "kicking",
            "mobattack", "lynching",

            # Crime / Illegal Activity
            "robbery", "robbed", "theft", "stealing", "stolen",
            "burglary", "looting", "vandalism", "vandalized",
            "breakin", "trespassing", "hijacking",

            # Weapons / Severe Injury
            "gun", "firearm", "pistol", "revolver",
            "knife", "blade", "machete", "weapon",
            "blood", "bleeding", "injured", "injury",
            "wounded", "unconscious", "collapsed", "dead", "death"
        }

        # Medium-risk indicators (context dependent)
        self.medium_risk_keywords = {
            # Suspicious Movement
            "running", "rushing", "chasing", "fleeing",
            "escaping", "sprinting", "dragging",

            # Aggressive / Panic Behavior
            "shouting", "yelling", "screaming",
            "arguing", "quarrel", "threatening",
            "panic", "chaos",

            # Crowd / Anomaly
            "crowd", "gathering", "mob",
            "surrounding", "blocking", "swarming",

            # Abnormal Posture / Condition
            "fallen", "falling", "lying", "collapsed",
            "slumped", "motionless",

            # Damage / Disorder
            "broken", "damaged", "shattered",
            "smashed", "cracked", "destroyed",
            "leaking", "spilled"
            }

        # Context amplifiers (boost meaning when combined)
        self.context_keywords =  {
                    # Roads & Public Areas
                    "road", "street", "highway", "junction",
                    "intersection", "crossing", "bridge",
                    "tunnel", "sidewalk",

                    # Vehicles
                    "vehicle", "car", "bike", "motorcycle",
                    "truck", "bus", "van", "auto", "scooter",

                    # Sensitive Places
                    "shop", "store", "bank", "atm",
                    "mall", "market", "warehouse",
                    "office", "factory",

                    # Buildings / Locations
                    "building", "house", "apartment",
                    "parking", "garage", "alley",
                    "lane", "corridor",

                     # Time / Visibility
                    "night", "dark", "midnight",
                    "evening", "lowlight", "foggy"
}

        # Override patterns → force DANGEROUS
        self.override_patterns =  [
                    ("fire", "smoke"),
                    ("fire", "building"),
                    ("crash", "injured"),
                    ("collision", "road"),
                    ("vehicle", "hit"),
                    ("blood", "ground"),
                    ("weapon", "person"),
                    ("knife", "person"),
                    ("gun", "person"),
                    ("unconscious", "road"),
                    ("lying", "road"),
                    ("collapsed", "street"),
                    ("mob", "violence"),
                    ("crowd", "panic"),
                ]

    # --------------------------------------------------

    def normalize_text(self, text: str) -> str:
        """Lowercase, remove punctuation, normalize spaces."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # --------------------------------------------------

    def detect(self, caption: str) -> Dict:
        """
        Analyze caption and return incident assessment.
        """
        normalized = self.normalize_text(caption)
        tokens = set(normalized.split())

        matched_high = self.high_risk_keywords & tokens
        matched_medium = self.medium_risk_keywords & tokens
        matched_context = self.context_keywords & tokens

        score = 0
        matched_signals: List[str] = []

        # --- Scoring ---
        if matched_high:
            score += 3 * len(matched_high)
            matched_signals.extend(list(matched_high))

        if matched_medium:
            score += 1 * len(matched_medium)
            matched_signals.extend(list(matched_medium))

        # Context keywords only count if some risk already exists
        if (matched_high or matched_medium) and matched_context:
            score += 1 * len(matched_context)
            matched_signals.extend(list(matched_context))

        # --- Override logic ---
        for a, b in self.override_patterns:
            if a in tokens and b in tokens:
                return self._dangerous_override(
                    caption,
                    matched_signals + [a, b]
                )

        # --- Final decision ---
        if score >= 4:
            level = "DANGEROUS"
        elif score >= 2:
            level = "SUSPICIOUS"
        else:
            level = "SAFE"

        explanation = self._build_explanation(level, score, matched_signals)

        return {
            "risk_level": level,
            "risk_score": score,
            "matched_signals": sorted(set(matched_signals)),
            "explanation": explanation
        }

    # --------------------------------------------------

    def _dangerous_override(self, caption: str, signals: List[str]) -> Dict:
        """Force dangerous classification for critical patterns."""
        return {
            "risk_level": "DANGEROUS",
            "risk_score": 999,
            "matched_signals": sorted(set(signals)),
            "explanation": (
                "Critical pattern detected. Immediate danger inferred "
                "from combined signals."
            )
        }

    def _build_explanation(self, level: str, score: int, signals: List[str]) -> str:
        if not signals:
            return "No significant risk indicators detected."
        return (
            f"Detected indicators: {', '.join(sorted(set(signals)))}. "
            f"Risk score = {score}. Classified as {level}."
        )
