"""Print the Gemini models your API key can use for generating text.

Usage:
    python list_models.py
Then copy a name (without the 'models/' prefix) into LLM_MODEL in config.py.
"""
import os
import sys

import requests

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    sys.exit("GEMINI_API_KEY is not set. Run: export GEMINI_API_KEY=your-key")

resp = requests.get(
    "https://generativelanguage.googleapis.com/v1beta/models",
    headers={"x-goog-api-key": api_key},
    params={"pageSize": 200},
    timeout=30,
)
if resp.status_code != 200:
    sys.exit(f"Error {resp.status_code}: {resp.text[:300]}")

for m in resp.json().get("models", []):
    if "generateContent" in m.get("supportedGenerationMethods", []):
        print(m["name"].removeprefix("models/"))
