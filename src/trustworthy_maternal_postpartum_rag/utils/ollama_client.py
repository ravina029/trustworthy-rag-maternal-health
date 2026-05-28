# src/trustworthy_maternal_postpartum_rag/utils/ollama_client.py

import json
import os
import subprocess
import urllib.request
from urllib.error import URLError, HTTPError


def call_ollama(prompt: str) -> str:
    """
    Calls Ollama locally.

    Uses Ollama's JSON mode only when the prompt explicitly requests JSON.
    Otherwise returns normal free text.
    """

    model = os.getenv("TMPRAG_OLLAMA_MODEL", "llama3")

    wants_json = (
        "valid json only" in prompt.lower()
        or "return json with exactly these keys" in prompt.lower()
        or "output must be valid json" in prompt.lower()
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
        },
    }

    if wants_json:
        payload["format"] = "json"

    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))

        return (data.get("response") or "").strip()

    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError):
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
