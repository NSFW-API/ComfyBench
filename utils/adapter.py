# utils/adapter.py
import json
import re
from openai import OpenAI
from typing import Any, Dict

from utils.model import COMPLETION_BASE_URL, COMPLETION_API_KEY


_JSON_RE = re.compile(r'\{.*\}', re.S)   # first {...} blob, greedy → safest

def _extract_json_block(text: str) -> Dict[str, Any]:
    """
    Pull the first {...} block out of `text` and json.loads it.
    Raises ValueError if nothing parses.
    """
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError("No JSON object found in adapter output.")
    return json.loads(m.group(0))


def json_adapter(raw_text: str, schema_str: str, model_name: str = "o3") -> Dict[str, Any]:
    """
    Convert `raw_text` to a dict that matches the schema
    (see original doc-string). Now returns python-dict 100 %.
    """
    client = OpenAI(
        base_url=COMPLETION_BASE_URL,
        api_key=COMPLETION_API_KEY,
    )

    usr_msg = (
        "You are a converter. Rewrite the user message as pure JSON that obeys "
        "the instructions you will receive. Do NOT add fields or commentary."
        "Return only JSON text."
        f"Original:\n'''{raw_text}'''\n\n"
        f"Instructions:\n{schema_str}\n\nOnly return JSON."
    )

    resp = client.responses.create(
        model=model_name,
        input=usr_msg,
        text={"format": {"type": "json_object"}},
    )

    # --- NEW robust parsing ---------------------------------
    text_out = getattr(resp, "output_text", None) or getattr(resp, "text", None)
    if text_out is None:
        # Fallback: extract from first message content
        text_out = resp.output[0].content[0].text   # last resort

    return _extract_json_block(text_out)
