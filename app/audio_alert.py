import os
import winsound


class AlertPlayer:
    def __init__(
        self,
        enabled=True,
        mode="beep",
        wav_path=None,
        trigger_level="Dangerous + Suspicious",
    ):
        self.enabled = enabled
        self.mode = mode
        self.wav_path = wav_path
        self.trigger_level = trigger_level

    def _should_play(self, risk_level):
        if risk_level == "SAFE":
            return False
        if self.trigger_level == "Dangerous only":
            return risk_level == "DANGEROUS"
        return risk_level in {"DANGEROUS", "SUSPICIOUS"}

    def play(self, risk_level):
        if not self.enabled or not self._should_play(risk_level):
            return

        if self.mode == "custom_wav" and self.wav_path and os.path.isfile(self.wav_path):
            winsound.PlaySound(
                self.wav_path,
                winsound.SND_FILENAME | winsound.SND_ASYNC,
            )
            return

        if risk_level == "DANGEROUS":
            winsound.Beep(2200, 300)
            winsound.Beep(1800, 450)
        elif risk_level == "SUSPICIOUS":
            winsound.Beep(1600, 250)
