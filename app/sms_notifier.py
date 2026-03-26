from __future__ import annotations

import base64
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional


class SMSNotifier:
    """
    Optional Twilio SMS sender controlled through environment variables.
    If configuration is missing, send() becomes a no-op and returns False.
    """

    def __init__(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        from_number: Optional[str] = None,
        to_number: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self.account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID", "")
        self.auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN", "")
        self.from_number = from_number or os.getenv("TWILIO_FROM_NUMBER", "")
        self.to_number = to_number or os.getenv("TWILIO_TO_NUMBER", "")
        self.enabled = enabled

    @property
    def is_configured(self) -> bool:
        return all(
            [
                self.enabled,
                self.account_sid,
                self.auth_token,
                self.from_number,
                self.to_number,
            ]
        )

    def send(self, body: str) -> bool:
        if not self.is_configured:
            return False

        url = (
            f"https://api.twilio.com/2010-04-01/Accounts/"
            f"{self.account_sid}/Messages.json"
        )
        payload = urllib.parse.urlencode(
            {
                "From": self.from_number,
                "To": self.to_number,
                "Body": body,
            }
        ).encode("utf-8")
        auth = base64.b64encode(
            f"{self.account_sid}:{self.auth_token}".encode("utf-8")
        ).decode("ascii")

        request = urllib.request.Request(url, data=payload, method="POST")
        request.add_header("Authorization", f"Basic {auth}")
        request.add_header("Content-Type", "application/x-www-form-urlencoded")

        try:
            with urllib.request.urlopen(request, timeout=10):
                return True
        except (urllib.error.URLError, urllib.error.HTTPError):
            return False
