"""
Takes an Evidence object (Day 6a) and sends it to a local LLM running
through Ollama at http://localhost:11434. The model only reasons from
the provided evidence and does not make up measurements, causes, or history.

Requires:
    1. Ollama installed and running (ollama.com/download)
    2. A downloaded model: ollama pull qwen2.5:7b-instruct
    3. pip install requests
"""

import json

import requests

from aegis_core.evidence_engine import Evidence

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"


SYSTEM_PROMPT = """You are AEGIS, an industrial reliability reasoning assistant.

You will receive a structured evidence object computed by statistical and \
machine learning tools (anomaly detection, change-point detection, and \
possibly a trained classifier with SHAP feature importance). You do NOT have \
access to raw sensor data, only this evidence summary.

CRITICAL RULES:
- Use ONLY the evidence provided. Do not invent sensor measurements, failure \
causes, probabilities, or maintenance history that isn't in the evidence.
- If the evidence is insufficient or ambiguous, explicitly say so instead of \
guessing.
- has_supervised_model tells you whether a trained fault classifier exists \
for THIS dataset. If it is false, model_probability and confidence will be \
null — this means the health_score is based ONLY on unsupervised anomaly \
detection. In that case you MUST say explicitly that there is no historical \
fault-labeled model backing this diagnosis, and that risk_level reflects \
statistical abnormality, not a learned failure probability.
- When has_supervised_model is true: risk_level (how abnormal it looks) and \
confidence_label (how reliable the conclusion is) are DIFFERENT things. A \
high risk_level with LOW confidence means "something looks wrong, but we're \
not sure why" — say that explicitly when it applies, don't collapse it into \
false certainty.
- top_features are the SHAP-derived drivers of the model's prediction (when \
available) — treat them as the "why", not something you get to reinterpret freely.

Respond with ONLY a JSON object (no markdown, no preamble) with exactly these keys:
{
  "machine_state": "one sentence, current state",
  "what_changed": "one or two sentences, referencing anomaly_score/change_point_nearby/degradation_trend",
  "likely_explanation": "one or two sentences hypothesis, grounded in top_features",
  "evidence_used": ["short bullet", "short bullet", ...],
  "confidence_statement": "one sentence; if has_supervised_model is false, explicitly say this is unsupervised-only and there is no learned failure probability backing it",
  "recommended_action": "one of: Monitor | Inspect | Schedule maintenance | Immediate attention",
  "action_detail": "one sentence justifying the recommended_action"
}
"""

REQUIRED_KEYS = [
    "machine_state", "what_changed", "likely_explanation", "evidence_used",
    "confidence_statement", "recommended_action", "action_detail",
]


def _extract_json(text: str) -> dict:
    """
    Local LLMs are not always consistent with the exact output format.
    This tries to recover the JSON if there is extra text or markdown around it.
    """
    text = text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first '{' and last '}' and try to parse that part.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return {"error": "Unable to parse JSON from the local LLM", "raw_response": text}


def call_ollama(system_prompt: str, user_message: str, model: str = DEFAULT_MODEL, host: str = OLLAMA_HOST) -> str:
    response = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "format": "json",  # Forces Ollama to return valid JSON.
            "stream": False,
            "options": {"temperature": 0.2},  # Low value for more consistent results.
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def aegis_reasoning(evidence: Evidence, model: str = DEFAULT_MODEL, retry_on_bad_json: bool = True) -> dict:
    user_message = (
        "Evidence:\n"
        + json.dumps(evidence.to_dict(), indent=2)
        + "\n\nGenerate the engineering diagnosis JSON as instructed."
    )

    raw_text = call_ollama(SYSTEM_PROMPT, user_message, model=model)
    result = _extract_json(raw_text)

    missing_keys = [k for k in REQUIRED_KEYS if k not in result]
    if missing_keys and retry_on_bad_json:
        # Try once more with a reminder before giving up.
        reminder = f"\n\nIMPORTANT: your previous response was missing keys: {missing_keys}. Return ONLY the complete JSON object."
        raw_text = call_ollama(SYSTEM_PROMPT, user_message + reminder, model=model)
        result = _extract_json(raw_text)

    return result


if __name__ == "__main__":
    fake_evidence = Evidence(
        timestamp="2020-07-15 15:00:00",
        health_score=45.2,
        risk_level="HIGH",
        anomaly_score=0.78,
        change_point_nearby=True,
        operating_regime="LOADED",
        top_features=["pressure_main", "h1", "temperature"],
        model_probability=0.81,
        confidence=0.42,
        confidence_label="MEDIUM",
        degradation_trend=0.65,
    )
    print("Calling Ollama at", OLLAMA_HOST, "with model", DEFAULT_MODEL, "...")
    result = aegis_reasoning(fake_evidence)
    print(json.dumps(result, indent=2, ensure_ascii=False))